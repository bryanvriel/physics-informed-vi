#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import h5py
import sys
import os

from hdflow import *
from mean_field_rheology import BATCH_SIZE, CHECKDIR, model_generator

def main():

    print(DTYPE)

    # Load prediction grid
    fname = 'profile.h5'
    with h5py.File(fname, 'r') as fid:
        x = fid['x'][()]

    x = np.linspace(x[0], x[-1], 64)
    
    # --------------------------------------------------------------------------------
    # Define functions and computational graph
    # --------------------------------------------------------------------------------

    # Create conditional normal distribution as mean field surrogate.
    # This will take in some inputs and generate mean field samples conditional on
    # those inputs, e.g.:
    #   z ~ dist.sample(x, y)
    surrogate_posterior = ConditionalNormal(model_generator, BATCH_SIZE)
    
    # --------------------------------------------------------------------------------
    # Run
    # --------------------------------------------------------------------------------

    # Create checkpoint manager and restore weights
    ckpt_manager = create_checkpoint_manager(
        CHECKDIR, restore=True, surrogate_posterior=surrogate_posterior
    )

    # Generate samples over batches
    print('Generating samples')
    n_batches = int(np.ceil(x.size / BATCH_SIZE))
    n_samples = 1000
    samples = np.zeros((n_samples, x.size))
    for b in tqdm(range(n_batches)):

        # Get batch of coordinates
        bslice = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
        xb = x[bslice]
        n_valid = xb.size
        # Optional extension to make batch size uniform
        if xb.size < BATCH_SIZE:
            n_extra = BATCH_SIZE - xb.size
            xb = np.hstack((xb, np.full((n_extra,), xb[-1])))

        # Generate samples for batch
        sb = surrogate_posterior.sample(n_samples, model_args=(xb.reshape(-1, 1),))
       
        # Store
        samples[:, bslice] = sb[:, :n_valid] 

    # Save to file
    with h5py.File('samples.h5', 'w') as fid:
        fid['samples'] = samples
        fid['x'] = x
    

if __name__ == '__main__':
    main()

# end of file
