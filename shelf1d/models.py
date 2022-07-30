#-*- coding: utf-8 -*-

import numpy as np
from hdflow import *
tfk = tfp.math.psd_kernels


class SurfaceNetwork(tf.keras.Model):
    """
    Feedforward neural network tasked with predicting surface velocity and thickness.
    """

    def __init__(self, norms, resnet=False, name='surf_net'):
        super().__init__(name=name)
        with tf.name_scope(name) as scope:
            if resnet:
                self.u_dense = ResidualNetwork(layer_sizes=[50, 50, 50, 50, 1], name='u_dense')
                self.h_dense = ResidualNetwork(layer_sizes=[50, 50, 50, 50, 1], name='h_dense')
            else:
                self.u_dense = DenseNetwork(layer_sizes=[50, 50, 50, 50, 1], name='u_dense')
                self.h_dense = DenseNetwork(layer_sizes=[50, 50, 50, 50, 1], name='h_dense')
            self.actfun = tf.tanh

    def call(self, An, training=False):
        u = self.u_dense(An, activation=self.actfun, training=training)
        h = self.h_dense(An, activation=self.actfun, training=training)
        return tf.concat(values=[u, h], axis=1)


class MeanNetwork(tf.keras.Model):
    """
    Feedforward neural network that predicts mean values for u, h, θ.
    """

    def  __init__(self, norms, resnet=False, name='mean_net'):
        super().__init__(name=name)
        with tf.name_scope(name) as scope:
            # Network for surface variables
            self.surf_dense = SurfaceNetwork(norms, resnet=resnet)
            # Tensors for scaling (un-normalizing) surface variables
            self.W_in, self.iW_in, self.b_in = assemble_scale_tensors(norms, ['x',])
            self.W_out, self.iW_out, self.b_out = assemble_scale_tensors(
                norms, ['u', 'h']
            )

    def call(self, x, training=False, inverse_norm=False):

        # Normalize inputs
        An = normalize_tensor(x, self.iW_in, self.b_in, pos=True)

        # Pass through networks
        params = self.surf_dense(An, training=training)

        # Optional inverse normalization
        if inverse_norm:
            params = inverse_normalize_tensor(params, self.W_out, self.b_out)

        # Unpack
        return unpack_variables(params)


def create_vgp(x, num_inducing_x, norms, jitter=1.0e-5, trainable_obs_variance=False, name='vgp'):
    """
    Instantiate a variational Gaussian Process with a fixed number of inducing points.

    Parameters
    ----------
    x: ndarray
        Array of x-coordinates used for training. Only used to get extents.
    num_inducing_x: int
        Number of inducing points.
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
            1.0e-8, tfb.Exp(), dtype=DTYPE, trainable=trainable_obs_variance,
            name='noise_variance'
        )

        # Initial array for inducing points (normalized coordinates)
        # Coordinate bounds are computed from data coordinates.
        x_ind = np.linspace(x.min(), x.max(), num_inducing_x)
        num_inducing = x_ind.size
        X_ind = x_ind.reshape(-1, 1)

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
        

def distance_prior(xb, L_scl, prior_std=1.0, return_dist=True):
    """
    Compute distance-dependent covariance matrix and instantiate a multivariate
    normal distribution with lower triangular linear operator.
    """
    # First compute distance matrix
    x = tf.squeeze(xb)
    dist_sq = tf.square(x[:, None] - x[None, :])

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

def unpack_variables(params):
    """
    Unpacks column tensors from concatenated tensor.
    """
    u = tf.expand_dims(params[:, 0], axis=1)
    h = tf.expand_dims(params[:, 1], axis=1)
    return u, h

# end of file
