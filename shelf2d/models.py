#-*- coding: utf-8 -*-

import numpy as np
from hdflow import *
import sys
tfk = tfp.math.psd_kernels


def unpack_variables(params):
    """
    Unpacks column tensors from concatenated tensor.
    """
    u = tf.expand_dims(params[:, 0], axis=1)
    v = tf.expand_dims(params[:, 1], axis=1)
    h = tf.expand_dims(params[:, 2], axis=1)
    B0 = tf.expand_dims(params[:, 3], axis=1)
    return u, v, h, B0


class MeanNetwork(tf.keras.Model):
    """
    Feedforward neural network that predicts mean values for u, h, θ, n.
    """

    def  __init__(self, norms, resnet=False, name='mean_net'):
        super().__init__(name=name)
        with tf.name_scope(name) as scope:

            # Network for surface variables and rheology prior
            if resnet:
                self.surf_dense = ResidualNetwork(
                    layer_sizes=[100, 100, 100, 100, 3], name='surf_dense'
                )
                self.B0_dense = ResidualNetwork(
                    layer_sizes=[50, 50, 50, 50, 1],
                    name='B0_dense'
                )
            else:
                self.surf_dense = DenseNetwork(
                    layer_sizes=[100, 100, 100, 100, 3], name='surf_dense'
                )
                self.B0_dense = DenseNetwork(
                    layer_sizes=[50, 50, 50, 50, 1],
                    name='B0_dense'
                )

            # Tensors for scaling (un-normalizing) surface variables
            self.W_in, self.iW_in, self.b_in = assemble_scale_tensors(norms, ['x', 'y'])
            self.W_out, self.iW_out, self.b_out = assemble_scale_tensors(
                norms, ['u', 'v', 'h', 'b']
            )

        self.actfun = tf.tanh

    def call(self, x, y, training=False, inverse_norm=False):

        # Normalize inputs
        A = tf.concat(values=[x, y], axis=1)
        An = normalize_tensor(A, self.iW_in, self.b_in)

        # Pass through networks
        surf_params = self.surf_dense(An, activation=self.actfun, training=training)
        B0 = self.B0_dense(An, activation=self.actfun, training=training)
        params = tf.concat(values=[surf_params, B0], axis=1)

        # Optional inverse normalization
        if inverse_norm:
            params = inverse_normalize_tensor(params, self.W_out, self.b_out)

        # Unpack
        return unpack_variables(params)


def create_vgp(x, y, num_inducing_x, num_inducing_y, norms, 
               jitter=1.0e-5, trainable_obs_variance=False, L_prior=10.0e3, name='vgp'):
    """
    Instantiate a variational Gaussian Process with a fixed number of inducing points.

    Parameters
    ----------
    x: ndarray
        Array of x-coordinates used for training. Only used to get extents.
    y: ndarray
        Array of y-coordinates used for training. Only used to get extents.
    num_inducing_x: int
        Initial number of inducing points in x-direction.
    num_inducing_y: int
        Initial Number of inducing points in y-direction.
    norms: dict of pgan.data.Normalizer
        Dict of normalization objects/parameters for input and output variables.
    jitter: float, optional
        Value added to diagonal of covariance matrix for numerical stability. Default: 1e-5.
    trainable_obs_variance: bool, optional
        Make observation noise variance trainable. Default: False.
    name: str, optional
        Model name. Default: 'vgp'.

    Returns
    -------
    vgp: tfd.VariationalGaussianProcess
        Configured variational Gaussian Process.
    """
    with tf.name_scope(name) as scope:
    
        # Create kernel with trainable parameters, and trainable observation noise
        # variance variable. Each of these is constrained to be positive using tf.Exp
        # bijector. Note that initial value corresponds to transformed output.
        amplitude = tfp.util.TransformedVariable(
            0.2, tfb.Exp(), dtype=DTYPE, name='amplitude'
        )
        length_scale = tfp.util.TransformedVariable(
            0.3, tfb.Exp(), dtype=DTYPE, name='length_scale'
        )
        noise_variance = tfp.util.TransformedVariable(
            1.0e-4, tfb.Exp(), dtype=DTYPE, trainable=trainable_obs_variance,
            name='noise_variance'
        )

        # Initial array for inducing points (normalized coordinates)
        x_ind = np.linspace(x.min(), x.max(), num_inducing_x)
        y_ind = np.linspace(y.min(), y.max(), num_inducing_y)
        x_ind, y_ind = [arr.ravel() for arr in np.meshgrid(x_ind, y_ind)]
        num_inducing = x_ind.size
        X_ind = np.column_stack((norms['x'](x_ind), norms['y'](y_ind)))

        # Ni
        inducing_index_points = tf.Variable(
            X_ind,
            dtype=DTYPE, name='inducing_index_points'
        )

        # Ni
        variational_inducing_loc = tf.Variable(
            np.zeros(num_inducing),
            dtype=DTYPE, name='variational_inducing_loc'
        )

        # Ni x Ni
        variational_inducing_scale = tf.Variable(
            np.eye(num_inducing),
            dtype=DTYPE, name='variational_inducing_scale'
        )

        # Create kernel
        kernel = tfk.ExponentiatedQuadratic(
            amplitude=amplitude,
            length_scale=length_scale
        )

        # Create VGP
        return tfd.VariationalGaussianProcess(
            kernel,
            index_points=X_ind,
            inducing_index_points=inducing_index_points,
            variational_inducing_observations_loc=variational_inducing_loc,
            variational_inducing_observations_scale=variational_inducing_scale,
            observation_noise_variance=noise_variance,
            jitter=jitter 
        )
        

def distance_prior(xb, yb, L_scl, prior_std=1.0, return_dist=True):
    """
    Compute distance-dependent covariance matrix and instantiate a multivariate
    normal distribution with lower triangular linear operator.
    """
    # First compute distance matrix
    x = tf.squeeze(xb)
    y = tf.squeeze(yb)
    dist_sq = tf.square(x[:, None] - x[None, :]) + \
              tf.square(y[:, None] - y[None, :])

    # Compute full covariance matrix
    cov = prior_std**2 * tf.exp(-dist_sq / (2.0 * L_scl**2))
    # Add jitter for numerical stability
    jitter = 1.0e-6 * tf.ones_like(x)
    cov = cov + tf.linalg.diag(jitter)

    # Mean of zero
    loc = tf.zeros_like(x)

    # Return mvn distribution or loc,cov directly
    if return_dist:
        return tfd.MultivariateNormalTriL(loc, tf.linalg.cholesky(cov))
    else:
        return loc, cov

def create_optimizer(optname, learning_rate=0.0005, **kwargs):
    """
    Convenience function for creating a tf.keras.optimizers.Optimizer.
    """
    if optname == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)
    elif optname == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.8, **kwargs)
    else:
        raise NotImplementedError('Unsupported optimizer.')

    return optimizer


# end of file
