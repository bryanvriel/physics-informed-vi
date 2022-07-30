#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import h5py
import sys
import os

from hdflow import *
from pgan.data import Data
from models import *
from utilities import *

usage = """
Usage: train_joint.py config.cfg
"""

def main(pars):

    # Make data objects
    _, data, norms = make_data(
        physics_batch_size=pars.train.batch_size, Np=1200
    )

    # Load full x-coordinate array so that VGP can compute spatial extents
    with h5py.File('data_noise.h5', 'r') as fid:
        xn_ref = norms['x'](fid['x'][()])

    # Make models
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet)
    vgp = create_vgp(xn_ref, pars.prior.num_inducing_x, norms,
                     trainable_obs_variance=pars.likelihood.trainable_obs_variance)

    # Create optimizer
    optimizer = create_optimizer(pars.pretrain.optimizer, learning_rate=pars.train.lr)

    # Load pretrained solution
    if not pars.train.restore:
        print('Loading pretrain weights from', pars.train.meannet_checkdir)
        create_checkpoint_manager(
            pars.train.meannet_checkdir,
            mean_net=mean_net,
            restore=True
        )

    # Create checkpoint manager for all models
    print('Checkpoint directory:', pars.train.checkdir)
    print('Restore:', pars.train.restore)
    ckpt_manager = create_checkpoint_manager(
        pars.train.checkdir,
        mean_net=mean_net,
        vgp=vgp,
        optimizer=optimizer,
        restore=pars.train.restore
    )

    # Initialize Gauss-Hermite quadrature sample points and weights
    quadrature_size = 3
    grid, weights = np.polynomial.hermite.hermgauss(quadrature_size)
    grid = grid.astype(np.float64)
    weights = tf.convert_to_tensor(weights.astype(np.float64), dtype=DTYPE)
    grid = tf.convert_to_tensor(grid, dtype=DTYPE)

    # Pre-construct some constants
    KL_SCALE = tf.constant(pars.prior.kl, dtype=DTYPE)
    PROB_SCALE = tf.constant(pars.train.prob_scale, dtype=DTYPE)
    TB_STD = tf.constant(pars.likelihood.std, dtype=DTYPE)
    n_exp = tf.constant(pars.prior.n_exp, dtype=DTYPE)

    # Define joint function
    @tf.function
    def loss_function(batch_physics):

        Nb = tf.cast(tf.shape(batch_physics.x)[0], dtype=DTYPE)

        # Run forward model to get drag residuals
        log_likely, index_points, _ = _forward_model(
            mean_net, vgp, n_exp, batch_physics.x, grid, weights, TB_STD
        )
        nll = -1 * log_likely

        # Construct prior at inducing points
        xn_inducing = vgp.inducing_index_points
        x_inducing = norms['x'].inverse(xn_inducing)
        prior = distance_prior(x_inducing, pars.prior.l_scale,
                               prior_std=pars.prior.std)
        
        # Variational mean and covariance at inducing points
        C_post = vgp.covariance(index_points=xn_inducing)
        θ_mean = vgp.mean(index_points=xn_inducing)
        post = tfd.MultivariateNormalTriL(θ_mean, tf.linalg.cholesky(C_post))

        # Compute (reverse) KL divergence
        kl = KL_SCALE / Nb * tfd.kl_divergence(post, prior)

        # Done
        return [PROB_SCALE * nll, PROB_SCALE * kl]

    # JIT and evaluate loss function in order to get access to trainable variables
    batch = data.train_batch()
    lv = loss_function(batch)
    trainable_variables = list(vgp.trainable_variables)
    print('\nTrainable variables:')
    for v in trainable_variables:
        print(v.name, v.shape)
    print('')
        
    # Run optimization
    train_vals, test_vals = train(
        trainable_variables, loss_function, data, optimizer, ckpt_manager,
        n_epochs=pars.train.n_epochs, clip=5, logfile=pars.train.logfile, ckpt_skip=5
    )


def _forward_model(mean_net, vgp, n_exp, x, grid, weights, Tb_std):

    # Physical constants
    rho_ice = tf.constant(917.0, dtype=DTYPE)
    rho_water = tf.constant(1024.0, dtype=DTYPE)
    g = tf.constant(9.80665, dtype=DTYPE)
    ε_0 = tf.constant(1.0e-5, dtype=DTYPE)
    B0 = tf.constant(400.0, dtype=DTYPE)
    W = tf.constant(30.0e3, dtype=DTYPE)
    Nb = tf.shape(x)[0]

    # Predict surface variables and un-normalize
    u, h = mean_net(x, inverse_norm=True)

    # Compute spatial gradients
    u_x = tf.gradients(u, x)[0]
    h_x = tf.gradients(h, x)[0]
    absu = tf.math.abs(u)
    usign = u / absu
    strain = tf.abs(u_x) + ε_0

    # Driving stress (kPa units)
    s_x = h_x * (1 - rho_ice / rho_water)
    Td = -1.0e-3 * rho_ice * g * h * s_x

    # Normalized index points
    Xn = normalize_tensor(x, mean_net.iW_in, mean_net.b_in, pos=True)

    # Pre-construct likelihood arrays
    Tb_loc = tf.zeros((Nb, 1), dtype=DTYPE)
    Tb_scale = Tb_std * tf.ones((Nb, 1), dtype=DTYPE)

    # Generate θ values using quadrature grid
    qf_loc = vgp.mean(index_points=Xn)
    qf_scale = vgp.stddev(index_points=Xn)
    qf_loc = _maybe_expand_dims_neg2(qf_loc)
    qf_scale = _maybe_expand_dims_neg2(qf_scale)
    weighted_ll = 0.0
    for i in range(3):

        # Compute θ at current quadrature grid point
        θ = np.sqrt(2.0) * qf_scale * grid[i] + qf_loc

        # Compute fluidity
        B = B0 * tf.exp(θ)

        # Effective viscosity
        eta = 0.5 * B * strain**((1.0 - n_exp) / n_exp)

        # Membrane stress
        Tm = 4 * tf.gradients(eta * h * u_x, x)[0]
       
        # Lateral drag
        Tl = 2 * usign * h / W * B * (5 * absu / W)**(1.0 / n_exp)

        # Drag residual
        Tb = Tm - Tl + Td

        # Compute weighted log-likelihood
        log_probs = tfd.Normal(loc=Tb_loc, scale=Tb_scale).log_prob(Tb)
        weighted_ll += weights[i] * tf.reduce_mean(log_probs)
        
    return weighted_ll, Xn, tf.squeeze(qf_loc)


def _maybe_expand_dims_neg2(a):
    """Inject a `1` into the shape of `a` only if `a` is non-scalar.
    Also handles the dynamic shape case.
    Args:
      a: a `Tensor`.
    Returns:
      maybe_expanded: a `Tensor` whose shape is unchanged if `a` was scalar,
      or, if `a` had shape `[..., A, B]`, a new `Tensor` which is the same as
      `a` but with shape `[..., A, 1, B]`.
    """
    if tf.TensorShape(a.shape).rank == 0:
        return a
    if tf.TensorShape(a.shape).rank is not None:
        return a[..., tf.newaxis]
    return tf.cond(tf.equal(tf.rank(a), 0),
                   lambda: a,
                   lambda: a[..., tf.newaxis])


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print(usage)
        sys.exit()
    pars = ParameterClass(args[0])
    main(pars)


# end of file
