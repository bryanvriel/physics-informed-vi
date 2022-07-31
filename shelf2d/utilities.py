#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pgan.data import Data, Normalizer, DataCollection
import configparser
import ast
import h5py
import sys
import os

H5FILE = 'data.h5'

# Uncertainty for icepack-estimated B
B0_STD = 5.0

def h5read(filename, dataset):
    if isinstance(dataset, str):
        with h5py.File(filename, 'r') as fid:
            data = fid[dataset][()]
        return data
    elif isinstance(dataset, (list, tuple)):
        data = []
        with h5py.File(filename, 'r') as fid:
            for key in dataset:
                data.append(fid[key][()])
        return data
    else:
        raise ValueError('Must provide dataset as str or list of str')

def compute_bounds(x, n_sigma=1.0, method='normal'):
    """
    Convenience function to compute lower/upper bounds for an array. Returns either:
        a) [μ - n * σ, μ + n * σ] (method = 'normal')
        b) [x_min, x_max]         (method = 'minmax')
    """
    if method == 'normal':
        mean = np.mean(x)
        std = np.std(x)
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        return [lower, upper]
    elif method == 'minmax':
        maxval = np.nanmax(x)
        minval = np.nanmin(x)
        return [minval, maxval]
    else:
        raise ValueError('Unsupported bounds determination method')

def make_normalizers():
    """
    Convenience function to create Normalizer objects for datasets in H5 file.
    """
    # Load data
    keys = ('x', 'y', 'u', 'v', 'h', 'b', 'u_err', 'v_err', 'h_err')
    arrs = [arr.astype(np.float64).reshape(-1, 1) for arr in h5read(H5FILE, keys)]
    x, y, u, v, h, b, u_err, v_err, h_err = arrs
    b_err = B0_STD * np.ones_like(u_err)
    
    # Normalize variables
    norms = {}
    for key, arr in zip(keys, arrs):
        vmin, vmax = compute_bounds(arr)
        norms[key] = Normalizer(vmin, vmax)

    return norms


def make_pretrain_data(seed=77, batch_size=256, preview=False, normalize_inputs=False,
                       np_dtype=np.float64):

    # Load data
    keys = ('x', 'y', 'u', 'v', 'h', 'b', 'u_err', 'v_err', 'h_err')
    arrs = [arr.astype(np_dtype).reshape(-1, 1) for arr in h5read(H5FILE, keys)]
    x, y, u, v, h, b, u_err, v_err, h_err = arrs
    b_err = B0_STD * np.ones_like(u_err)
    
    # Make normalizers
    norms = make_normalizers()

    # Normalize output variables
    u = norms['u'](u)
    v = norms['v'](v)
    h = norms['h'](h)
    b = norms['b'](b)
    u_err = norms['u'].forward_scale(u_err)
    v_err = norms['v'].forward_scale(v_err)
    h_err = norms['h'].forward_scale(h_err)
    b_err = norms['b'].forward_scale(b_err)

    if normalize_inputs:
        x = norms['x'](x)
        y = norms['y'](y)

    if preview:

        def _scatter(ax, x, y, z):
            sc = ax.scatter(x, y, s=20, c=z, cmap='turbo')
            plt.colorbar(sc, ax=ax)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axes.ravel()
        _scatter(ax1, x, y, u)
        _scatter(ax2, x, y, v)
        _scatter(ax3, x, y, h)
        _scatter(ax4, x, y, b)
        plt.tight_layout()
        plt.show()

        sys.exit()

    # Create data object
    data = Data(train_fraction=0.85,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                full_traversal=False,
                x=x, y=y, u=u, v=v, h=h, b=b,
                u_err=u_err, v_err=v_err, h_err=h_err, b_err=b_err)
    
    return data, norms


def make_rheology_data(filename=H5FILE, seed=60, batch_size=128, preview=False,
                       np_dtype=np.float64):

    # Load data
    keys = ('xp', 'yp')
    arrs = [arr.astype(np_dtype).reshape(-1, 1) for arr in h5read(H5FILE, keys)]
    xp, yp = arrs
    
    # Make normalizers
    norms = make_normalizers()
    
    if preview:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(xp.squeeze(), yp.squeeze(), 'o')
        plt.tight_layout()
        plt.show()
        sys.exit()

    # Create data object
    data = Data(train_fraction=0.85,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                full_traversal=True,
                x=xp, y=yp)
                
    return data, norms


default_cfg = """
[prior]
l_scale = 10.0e3
std = 0.1
kl = 1.0
n_exp = 4.0
num_inducing_x = 15
num_inducing_y = 10

[likelihood]
std = 0.1
trainable_obs_variance = False

[train]
lr = 0.0002
restore = False
n_epochs = 1000
batch_size = 512
prob_scale = 0.1
meannet_checkdir = 'checkpoints/checkpoints_pretrain'
checkdir = 'checkpoints/checkpoints_joint'
logfile = 'log_joint'
optimizer = 'adam'

[pretrain]
lr = 0.0002
restore = False
n_epochs = 1000
batch_size = 512
checkdir = 'checkpoints/checkpoints_pretrain'
logfile = 'log_pretrain'
resnet = False
optimizer = 'adam'"""

class GenericClass:
    pass

class ParameterClass:

    def __init__(self, cfgfile=None):

        # Parse default configuration
        cfg = configparser.ConfigParser()
        cfg.read_string(default_cfg)
        for section in ('prior', 'likelihood', 'train', 'pretrain'):
            sub_pars = GenericClass()
            for key, value in cfg[section].items():
                value = ast.literal_eval(value)
                setattr(sub_pars, key, value)
            setattr(self, section, sub_pars)

        # If a file is provided, parse and update values
        if cfgfile is not None:
            cfg = configparser.ConfigParser()
            cfg.read(cfgfile)
            for section in ('prior', 'likelihood', 'train', 'pretrain'):
                sub_pars = getattr(self, section)
                for key, value in cfg[section].items():
                    value = ast.literal_eval(value)
                    setattr(sub_pars, key, value)
                setattr(self, section, sub_pars)


# end of file
