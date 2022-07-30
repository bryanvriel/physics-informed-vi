#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pgan.data import Data, Normalizer, DataCollection
import configparser
import ast
import h5py
import sys
import os

H5FILE = 'data_noise.h5'
DKEYS = ['x', 'u', 'h']

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
    # Read data and scale down to km
    with h5py.File(H5FILE, 'r') as fid:
        x, u, h = [fid[key][()] for key in DKEYS]
    keys = ['x', 'u', 'h']
    arrs = [x, u, h]

    # Normalize variables
    norms = {}
    for key, arr in zip(keys, arrs):
        if key in ('x',):
            vmin, vmax = compute_bounds(arr, method='minmax')
            norms[key] = Normalizer(vmin, vmax, pos=True) # inputs normalized to [0, 1]
        else:
            vmin, vmax = compute_bounds(arr)
            norms[key] = Normalizer(vmin, vmax)

    return norms


def make_data(seed=70, batch_size=40, physics_batch_size=256, train_fraction=0.85, Np=1000):
    """
    Create data object for surface variables and coordinates for physics-based losses.

    Parameters
    ----------
    seed: int, optional
        Integer for random seed. Default: 70.
    batch_size: int, optional
        Batch size. Default: 40.
    train_fraction: float, optional
        Fraction of data to use for training. Default: 0.85.
    Np: int, optional
        Number of coordinates for physics-based losses. Default: 1000.

    Returns
    -------
    data_surface: pgan.data.Data
        Data object for surface variables.
    data_physics: pgan.data.Data
        Data object for physics-based losses.
    norms: dict of pgan.data.Normalizer
        Dictionary of normalization objects.
    """
    # Read data and scale down to km
    keys = DKEYS + ['u_err', 'h_err']
    with h5py.File(H5FILE, 'r') as fid:
        x, u, h, u_err, h_err = [fid[key][()].reshape(-1, 1) for key in keys]

    # Make normalizers
    norms = make_normalizers()

    # Normalize output variables
    u = norms['u'](u)
    h = norms['h'](h)
    # Also normalize sigmas (data errors)
    u_err = norms['u'].forward_scale(u_err)
    h_err = norms['h'].forward_scale(h_err)
    
    # Make random coordinates for physics-based loss function
    rng = np.random.default_rng(seed)
    x_min, x_max = np.min(x), np.max(x)
    #xp = (x_max - x_min) * rng.random(Np) + x_min

    x_choice = np.linspace(x_min, x_max, 10000)
    p = np.linspace(0.3, 1.0, x_choice.size)
    p /= np.sum(p)
    xp = rng.choice(x_choice, p=p, size=Np, replace=False)
    #plt.hist(xp); plt.show(); sys.exit()

    xp = xp.reshape(-1, 1)

    # Make data object for surface variables
    data_surface = Data(train_fraction=train_fraction,
                        batch_size=batch_size,
                        shuffle=True,
                        seed=seed,
                        full_traversal=True,
                        x=x, u=u, h=h, u_err=u_err, h_err=h_err)

    # Make data object for physics losses
    data_physics = Data(train_fraction=train_fraction,
                        batch_size=physics_batch_size,
                        shuffle=True,
                        seed=seed,
                        full_traversal=True,
                        x=xp)

    return data_surface, data_physics, norms

# Default configuration string. If these aren't set in an external file, the
# default values are used.
default_cfg = """
[prior]
l_scale = 10.0e3
std = 0.1
kl = 1.0
n_exp = 3.0
num_inducing_x = 15

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
checkdir = 'checkpoints/checkpoints'
logfile = 'log_train'
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
    """
    Class for storing processing configuration parameters, either from default or
    a separate configuration file.
    """

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
