#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import h5py
import sys
import os

from geometry import *

def main():

    # Load samples
    with h5py.File('samples.h5', 'r') as fid:
        x = fid['x'][()]
        samples = fid['samples'][()]
        count = np.arange(samples.shape[0])
    
    # Covariance matrix
    C = np.cov(samples, rowvar=False)

    plt.imshow(C, interpolation='nearest', cmap='RdBu_r', clim=(-0.25, 0.25))
    plt.colorbar()
    plt.show()

    # Initialize profile
    profile = load_profile_from_h5('profile.h5')

    # Random traces
    rng = np.random.default_rng(80)
    inds = rng.choice(x.size, size=8, replace=False)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 8))
    for cnt, ind in enumerate(inds):
        ax = axes.ravel()[cnt]
        ax.plot(count, samples[:, ind])
    plt.tight_layout()
    plt.show()

    # Stats
    theta_mean = np.median(samples, axis=0)
    theta_std = np.std(samples, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    
    B_ref = (1.0e9 * profile.A)**(-1.0 / 3.0)
    logB_ref = np.log(B_ref)
    
    ax.plot(profile.x, logB_ref)

    logB = 5.85 + theta_mean
    line, = ax.plot(x, logB)
    ax.fill_between(x, logB - theta_std, logB + theta_std, alpha=0.4, color=line.get_color())

    fig.set_tight_layout(True)
    plt.show()


def main_predictions():

    from VolcanicSource import Mogi

    # Load data
    with h5py.File('../data.h5', 'r') as fid:
    
        # Load grids
        x, y, U, V, W = [fid[key][()] for key in ('x', 'y', 'u', 'v', 'w')]
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        theta_ref = fid['theta'][()]

    # Load samples
    with h5py.File('samples.h5', 'r') as fid:
        samples = fid['samples'][()][HEAD:, :]
    theta = np.mean(samples, axis=0)
    #theta = theta_ref
   
    # Two mogi sources
    x1, y1, z1, V1 = theta[:4]
    x2, y2, z2, V2 = theta[4:]
    
    # Create objects
    m1 = Mogi(x0=x1, y0=y1, z0=z1, dV=V1)
    m2 = Mogi(x0=x2, y0=y2, z0=z2, dV=V2)
    
    # Compute sum displacement
    u1, v1, w1 = m1.computeDisplacement(X, Y, Z)
    u2, v2, w2 = m2.computeDisplacement(X, Y, Z)
    u = 1.0e3 * (u1 + u2)
    v = 1.0e3 * (v1 + v2)
    w = 1.0e3 * (w1 + w2)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))

    def imshow(i, obs, pred):
        ax1, ax2 = axes[i, :]
        im1 = ax1.imshow(obs, cmap='turbo')
        im2 = ax2.imshow(pred, cmap='turbo', clim=im1.get_clim())

    imshow(0, U, u)
    imshow(1, V, v)
    imshow(2, W, w)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
    #main_predictions()

# end of file
