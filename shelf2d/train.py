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
Usage: train_joint.py run.cfg
"""

np_dtype = np.float64

def main(pars):

    # Make data objects
    data_pre, norms = make_pretrain_data(preview=False, normalize_inputs=False,
                                         batch_size=pars.train.batch_size,
                                         np_dtype=np_dtype)
    data_drag, _ = make_rheology_data(preview=False, batch_size=pars.train.batch_size,
                                      np_dtype=np_dtype)
    data = DataCollection(data_pre, data_drag)

    # Reference coordinates
    with h5py.File(H5FILE, 'r') as fid:
        x_ref = fid['x'][()]
        y_ref = fid['y'][()]

    # Make models
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet)
    vgp = create_vgp(x_ref, y_ref, pars.prior.num_inducing_x, pars.prior.num_inducing_y, norms)

    # Create optimizer
    optimizer = create_optimizer(pars.train.optimizer, learning_rate=pars.pretrain.lr)

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

    @tf.function
    def loss_function(batch):

        # Unpack batch
        batch_pre, batch_drag = batch
        Nb = tf.cast(tf.shape(batch_drag.x)[0], dtype=DTYPE)

        # Predict at data coordinates (for computing MSE loss)
        up, vp, hp, _ = mean_net(batch_pre.x, batch_pre.y, inverse_norm=False)

        # Compute negative log-likelihood
        data_nll = tf.reduce_mean(
            (tf.square(up - batch_pre.u) / tf.square(batch_pre.u_err)) +
            (tf.square(vp - batch_pre.v) / tf.square(batch_pre.v_err)) +
            (tf.square(hp - batch_pre.h) / tf.square(batch_pre.h_err))
        )

        # Run forward model to get drag residuals
        log_likely, index_points, θ_mean = _forward_model(
            mean_net, vgp, n_exp, batch_drag.x, batch_drag.y, grid, weights, TB_STD
        )
        nll = -1 * log_likely
        
        # Construct prior
        prior = distance_prior(batch_drag.x, batch_drag.y, pars.prior.l_scale,
                               prior_std=pars.prior.std)

        # Variational mean and covariance
        C_post = vgp.covariance(index_points=index_points)
        post = tfd.MultivariateNormalTriL(θ_mean, tf.linalg.cholesky(C_post))

        # Compute (reverse) KL divergence
        kl = KL_SCALE / Nb * tfd.kl_divergence(post, prior)

        # Done
        return [data_nll, PROB_SCALE * nll, PROB_SCALE * kl]

    # JIT and evaluate loss function in order to get access to trainable variables
    batch = data.train_batch()
    lv = loss_function(batch)
    trainable_variables = list(vgp.trainable_variables)
    trainable_variables += mean_net.surf_dense.trainable_variables
    print('\nTrainable variables:')
    for v in trainable_variables:
        print(v.name, v.shape)
    print('')
        
    # Run optimization
    train_vals, test_vals = train(
        trainable_variables, loss_function, data, optimizer, ckpt_manager,
        n_epochs=pars.train.n_epochs, clip=5, logfile=pars.train.logfile, ckpt_skip=10
    )


def _forward_model(mean_net, vgp, n_exp, x, y, grid, weights, Tb_std):
    """
    Computes basal drag residual negative log-likelihood. Uses the 2D SSA momentum balance.
    """
    # Physical constants
    rho_ice = tf.constant(917.0, dtype=DTYPE)
    rho_water = tf.constant(1024.0, dtype=DTYPE)
    g = tf.constant(9.80665, dtype=DTYPE)
    ε_0 = tf.constant(1.0e-5, dtype=DTYPE)
    Nb = tf.shape(x)[0]

    # Predict surface variables
    u, v, h, B0 = mean_net(x, y, inverse_norm=True)

    # Compute spatial gradients
    u_x = tf.gradients(u, x)[0]
    v_x = tf.gradients(v, x)[0]
    h_x = tf.gradients(h, x)[0]
    u_y = tf.gradients(u, y)[0]
    v_y = tf.gradients(v, y)[0]
    h_y = tf.gradients(h, y)[0]

    # Driving stress (kPa units)
    s_x = h_x * (1 - rho_ice / rho_water)
    s_y = h_y * (1 - rho_ice / rho_water)
    tdx = -1.0e-3 * rho_ice * g * h * s_x
    tdy = -1.0e-3 * rho_ice * g * h * s_y

    # Normalized index points
    X = tf.concat(values=[x, y], axis=1)
    Xn = normalize_tensor(X, mean_net.iW_in, mean_net.b_in)

    # Pre-construct likelihood arrays
    Tb_loc = tf.zeros((2*Nb, 1), dtype=DTYPE)
    Tb_scale = Tb_std * tf.ones((2*Nb, 1), dtype=DTYPE)

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
        strain = tf.sqrt(u_x**2 + v_y**2 + 0.25 * (u_y + v_x)**2 + u_x * v_y) + ε_0
        eta = 0.5 * B * strain**((1.0 - n_exp) / n_exp)

        # Membrane stress arguments
        tmx_argx = 2.0 * eta * h * (2.0 * u_x + v_y)
        tmy_argy = 2.0 * eta * h * (2.0 * v_y + u_x)
        tmxy_arg = eta * h * (u_y + v_x)

        # Membrane stresses
        tmx = tf.gradients(tmx_argx, x)[0] + tf.gradients(tmxy_arg, y)[0]
        tmy = tf.gradients(tmy_argy, y)[0] + tf.gradients(tmxy_arg, x)[0]

        # Compute drag mean predictions (this should nominally be zero for ice shelves)
        tbx = tmx + tdx
        tby = tmy + tdy
        Tb = tf.concat(values=[tbx, tby], axis=0)

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
