#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

# Import hdflow and tensorflow dependencies
from hdflow import *

# Processing parameters
VAR_KEYS = ['u', 'u_x', 'u_xx', 'en', 'en_x', 'h', 'h_x']
N_EPOCHS = 3000
BATCH_SIZE = 20
LEARNING_RATE = 0.0002
CHECKDIR = 'checkpoints_rheology'

def main():

    # Load data
    train_ds, test_ds = load_data('data.h5', batch_size=BATCH_SIZE, verbose=True)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # --------------------------------------------------------------------------------
    # Define loss function and its components
    # --------------------------------------------------------------------------------

    # Constant prior with fixed size
    bshape = (BATCH_SIZE,)
    prior = tfd.MultivariateNormalDiag(loc=tf.zeros(bshape, dtype=DTYPE),
                                       scale_diag=tf.ones(bshape, dtype=DTYPE))
    
    # Create conditional normal distribution as mean field surrogate.
    # This will take in some inputs and generate mean field samples conditional on
    # those inputs, e.g.:
    #   z ~ dist.sample(x, y)
    surrogate_posterior = ConditionalNormal(model_generator, BATCH_SIZE)
        
    @tf.function
    def loss_function(batch, n_samples=5):
        """
        Evaluate loss function on batch of data and return list of losses. Here, our target
        data values are computed from the momentum balance residual of the Shallow Shelf
        Approximation (SSA). The target values are used to construct the posterior distribution.
        """
        # Unpack the batch
        x, u, u_x, u_xx, en, en_x, h, h_x = batch
        # Squeeze surface variables
        u, u_x, u_xx, en, en_x, h, h_x = [
            tf.squeeze(arr) for arr in
            (u, u_x, u_xx, en, en_x, h, h_x)
        ]

        # Physical constant for converting posterior sample to physical quantity B
        logB0 = tf.constant(5.85, dtype=DTYPE)
 
        # Specify target basal drag, which is nominally zero
        Tb_obs = tf.zeros_like(u)
        Tb_scale = 0.5 * tf.ones_like(u)

        # Compute Monte Carlo expectation of KL for finite set of samples
        KL = 0.0
        for s in range(n_samples):

            # Generate sample from surrogate
            θ = surrogate_posterior.sample(1, model_args=(x,))
            θs = tf.squeeze(θ)

            # Evaluate ice stiffness parameter B here (needs to be at this level
            # in order to compute tf.gradients()
            B = tf.exp(logB0 + θs)
            B_x = tf.squeeze(tf.gradients(B, x)[0])

            # Evaluate physics model (SSA)
            Tb = compute_basal_drag(B, B_x, x, u, u_x, u_xx, en, en_x, h, h_x)
            
            # Instanate an MVN diag likelihood
            Tb_dist = tfd.MultivariateNormalDiag(loc=Tb, scale_diag=Tb_scale)

            # Evaluate total target posterior log probability
            f_target = prior.log_prob(θ) + Tb_dist.log_prob(Tb_obs)
            
            # Evaluate surrogate log probability
            f_surrogate = surrogate_posterior.log_prob(θ, model_args=(x,))

            # Compute reverse KL-divergence
            logu = f_target - f_surrogate
            KL += tfp.vi.kl_reverse(logu)

        # Final variational loss is simply the mean KL
        var_loss = KL[0] / n_samples
        
        return [var_loss,]

    # JIT and evaluate loss function in order to get access to trainable variables
    batch = next(iter(test_ds))
    loss_function(batch)
    trainable_variables = surrogate_posterior.trainable_variables

    # Create checkpoint manager
    ckpt_manager = create_checkpoint_manager(
        CHECKDIR, surrogate_posterior=surrogate_posterior
    )

    # Run training
    train(trainable_variables, loss_function, train_ds, test_ds, optimizer,
          ckpt_manager, n_epochs=N_EPOCHS)


@tf.function
def compute_basal_drag(B, B_x, x, u, u_x, u_xx, en, en_x, h, h_x):
    """
    Evaluate Shallow Shelf Approximation (SSA) momentum balance to compute basal
    drag values, which should be nominally zero.
    """
    # Physical constants
    rho_ice = tf.constant(917.0, dtype=DTYPE)
    rho_water = tf.constant(1024.0, dtype=DTYPE)
    g = tf.constant(9.80665, dtype=DTYPE)
    logB0 = tf.constant(5.85, dtype=DTYPE)
    ε_0 = tf.constant(1.0e-5, dtype=DTYPE)
    W = tf.constant(30.0e3, dtype=DTYPE)
    n_exp = tf.constant(3.0, dtype=DTYPE)
    
    # Effective viscosity
    strain = tf.abs(u_x) + ε_0
    eta = 0.5 * B * en
    eta_x = 0.5 * (B_x * en + B * en_x)

    # Longitudinal stress
    Tm = 4.0 * (eta_x * h * u_x +
                eta * h_x * u_x +
                eta * h * u_xx)

    # Lateral drag
    absu = tf.math.abs(u)
    usign = u / absu
    Tl = 2 * usign * h / W * B * (5 * absu / W)**(1.0 / n_exp)

    # Driving stress
    s_x = h_x * (1 - rho_ice / rho_water)
    Td = -1.0e-3 * rho_ice * g * h * s_x

    # Compute drag mean predictions (this should nominally be zero for ice shelves)
    Tb = Tm - Tl + Td

    return Tb

class ParameterModel(tf.keras.Model):
    """
    Feedforward neural network that predicts parameters for conditonal normal distribution.
    """

    def  __init__(self, x_min, x_max, name='param_net'):
        # Initialize parent class
        super().__init__(name=name)
    
        # Create dense bayesian network to output loc and scale (i.e., mean and stddev)
        self.dense = DenseNetwork(layer_sizes=[50, 50, 50, 2], name='dense')

        # Store bounds for input variable
        self.x_min = tf.constant(x_min, dtype=tf.float64)
        self.x_max = tf.constant(x_max, dtype=tf.float64)
    
        return
    
    def call(self, x, dx=0.0, squeeze=False, training=False):
        """
        Input space coordinates x and optional (post-normalized) perturbation dx.
        Output loc and scale for independent normal distribution.
        """
        # Normalize inputs and add any perturbations
        xn = (x - self.x_min) / (self.x_max - self.x_min) + dx

        # Pass through network
        params = self.dense(xn, training=training)

        # Unpack loc and scale
        loc = tf.expand_dims(params[:, 0], axis=1)
        scale = 1.0e-3 + tf.nn.softplus(tf.expand_dims(params[:, 1], axis=1))

        # Optional squeeze
        if squeeze:
            loc = tf.squeeze(loc)
            scale = tf.squeeze(scale)

        return loc, scale

def model_generator():
    """
    Generator function to instantiate a ParameterModel. Useful for instantiation
    within a tf.name_scope.
    """
    # Get bounds of input coordinates
    with h5py.File('data.h5', 'r') as fid:
        x = fid['x'][()]
    x_min = np.min(x)
    x_max = np.max(x)

    # Instantiate and return model
    return ParameterModel(x_min, x_max)

def load_data(filename, seed=24, split_seed=30, batch_size=64, train_fraction=0.85,
              verbose=False):
    """
    Read dataset from HDF5 file and create train and test tf.data.Datasets.
    """
    # Load raw data into dict
    with h5py.File(filename, 'r') as fid:
        np_data = [fid['x'][()]]
        for key in VAR_KEYS:
            np_data.append(fid[key][()])
        N = fid['x'].shape[0]

    # Create full dataset and shuffle
    full_ds = (tf.data.Dataset.from_tensor_slices(tuple(np_data))
                              .shuffle(N, seed=split_seed))

    # Get train/test split parameters
    N_train = int(train_fraction * N)
    N_test = N - N_train
    n_batches = int(np.ceil(N_train / batch_size))
    if verbose:
        print('Number of training examples:', N_train)
        print('Number of testing examples:', N_test)
        print('Number of batches:', n_batches)

    # Create training set
    # Subset -> shuffle -> repeat indefinitely -> extract batch -> limit number of batches
    train_ds = (full_ds.take(N_train)
                       .shuffle(N, seed=seed, reshuffle_each_iteration=True)
                       .repeat()
                       .batch(batch_size)
                       .take(n_batches))

    # Create test set
    # Subset -> shuffle -> repeat indefinitely -> extract batch
    test_ds = (full_ds.skip(N_train)
                      .shuffle(N, seed=seed, reshuffle_each_iteration=True)
                      .repeat()
                      .batch(min(batch_size, N_test)))
    
    return train_ds, test_ds

if __name__ == '__main__':
    main()

# end of file
