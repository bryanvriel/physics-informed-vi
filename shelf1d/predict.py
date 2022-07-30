#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import h5py
import sys
import os

from hdflow import *
from models import *
from utilities import *

mosaic = """
AB
CB
"""

def main():

    pars = ParameterClass('run.cfg')

    # Load full coordinates
    with h5py.File('data_noise.h5', 'r') as fid:
        x, u_obs, h_obs, u_noisy, h_noisy, B_ref = [
            fid[key][()] for key in
            ('x', 'u_ref', 'h_ref', 'u_noisy', 'h_noisy', 'B_ref')
        ]
        u_ref = fid['u_ref'][()]
        strain_ref = np.gradient(u_ref, x, edge_order=2)
        theta_ref = np.log(B_ref) - np.log(400.0)
    x = x.reshape(-1, 1)

    # Load normalizers used for training
    _, _, norms = make_data()
    
    # Make models
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet)
    vgp = create_vgp(x, pars.prior.num_inducing_x, norms,
                     trainable_obs_variance=pars.likelihood.trainable_obs_variance)

    use_pretrain = False
    if use_pretrain:
        CHECKDIR = pars.pretrain.checkdir
    else:
        CHECKDIR = pars.train.checkdir
    
    # Create checkpoint manager
    print('Loading weights from', CHECKDIR)
    ckpt_manager = create_checkpoint_manager(
        CHECKDIR,
        mean_net=mean_net,
        vgp=vgp,
        restore=True
    )
    print_gp_summary(vgp)

    # Predict surface variables
    u, h = [arr.numpy().squeeze() for arr in mean_net(x, inverse_norm=True)]
    strain = np.gradient(u, x.squeeze(), edge_order=2)
    with h5py.File('output_predictions.h5', 'w') as fid:
        fid['x'] = x.squeeze()
        fid['u'] = u
        fid['h'] = h

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 7))
    ax1, ax2 = axes[:, 0]
    ax3, ax4 = axes[:, 1]
    
    ax1.plot(x, u_obs, 'k', lw=3)
    ax1.plot(x, u_noisy, lw=0.5)
    line, = ax1.plot(x, u, lw=2)

    ax2.plot(x, h_obs, 'k', lw=3)
    ax2.plot(x, h_noisy, lw=0.5)
    line, = ax2.plot(x, h, lw=2)

    ax3.plot(x, strain_ref, 'k', lw=3)
    ax3.plot(x, strain, lw=2)

    ax4.plot(x, np.gradient(h_obs, x.squeeze(), edge_order=2), 'k', lw=3)
    ax4.plot(x, np.gradient(h, x.squeeze(), edge_order=2), lw=2)

    plt.tight_layout()
    if use_pretrain:
        plt.show()
        return

    # Get posterior mean and covariance
    rng = np.random.default_rng(100)
    Xn = normalize_tensor(x, mean_net.iW_in, mean_net.b_in, pos=True)
    θ_mean = vgp.mean(index_points=Xn).numpy().squeeze()
    θ_cov = vgp.covariance(index_points=Xn).numpy()

    # Generate mvn samples
    θ_samples = rng.multivariate_normal(θ_mean, θ_cov, size=3000)
    # Convert to B
    B_samples = 400 * np.exp(θ_samples)

    # Compute stochastic drag predictions
    Tb = forward_model(x.squeeze(), u, h, B_samples)
    print('Drag std:', np.std(Tb))

    # Save samples to file
    x = x.squeeze()
    with h5py.File('samples_vgp.h5', 'w') as fid:
        fid['x'] = x
        fid['theta_samples'] = θ_samples
        fid['B_samples'] = B_samples

    fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 5))

    # Plot θ samples 
    axes['A'].plot(x, θ_samples.T, color='C0', lw=0.6, alpha=0.6)
    axes['A'].plot(x, theta_ref, 'k', lw=3)
    axes['A'].set_ylim(-2, 1)

    # Plot stochastic drag predictions
    axes['C'].plot(x, Tb.T, color='C0', lw=0.6, alpha=0.6)
    axes['C'].axhline(0.0, color='k', lw=3, ls='-')
    axes['C'].set_ylim(-7, 7)
    axes['C'].grid(True, ls=':', lw=0.6)

    # Show covariance matrix for θ
    im = axes['B'].imshow(θ_cov, cmap='RdBu_r', clim=(-0.1, 0.1))
    plt.colorbar(im, ax=axes['B'])

    plt.tight_layout()
    plt.show()


def forward_model(x, u, h, B):
    """
    Predict basal drag residuals for samples of rigidity, B.
    """
    # Physical constants
    rho_ice = 917.0
    rho_water = 1024.0
    g = 9.80665
    ε_0 = 1.0e-5
    W = 30.0e3
    n_exp = 3.0

    # Compute spatial gradients
    u_x = np.gradient(u, x, edge_order=2)
    h_x = np.gradient(h, x, edge_order=2)

    # Effective viscosity
    strain = np.abs(u_x) + ε_0
    eta = 0.5 * B * strain**((1.0 - n_exp) / n_exp)

    # Membrane stress
    Tm = 4 * np.gradient(eta * h * u_x, x, axis=1, edge_order=2)

    # Lateral drag
    absu = np.abs(u)
    usign = u / absu
    Tl = 2 * usign * h / W * B * (5 * absu / W)**(1.0 / n_exp)

    # Driving stress
    s_x = h_x * (1 - rho_ice / rho_water)
    Td = -1.0e-3 * rho_ice * g * h * s_x

    # Compute drag mean predictions (this should nominally be zero for ice shelves)
    Tb = Tm - Tl + Td

    return Tb


def print_gp_summary(gp):
    print('Variable values for', gp.name)
    print(' - amplitude:', gp.kernel.base_kernel.amplitude.numpy())
    print(' - length scale:', gp.kernel.base_kernel.length_scale.numpy())
    print(' - noise variance:', gp.observation_noise_variance.numpy())
    print(' - inducing points:', gp.inducing_index_points.numpy().squeeze())
    print('')


if __name__ == '__main__':
    main()

# end of file
