#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from tqdm import tqdm
import argparse
import sys

def parse_command_line():
    parser = argparse.ArgumentParser(description="""
        Plot training lots.""")
    parser.add_argument('logfile', type=str, help='Input log file.')
    parser.add_argument('-head', action='store', type=int, default=0,
        help='Skip header lines.')
    parser.add_argument('-skip', action='store', type=int, default=1,
        help='Decimation factor. Default: 1.')
    return parser.parse_args()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def main(args):

    # Load log
    epochs, losses = read_log(args.logfile)
    nvar = losses.shape[1] // 2

    # Chop off header
    epochs = epochs[args.head::args.skip]
    losses = losses[args.head::args.skip]

    # Median filter
    #for j in tqdm(range(losses.shape[1])):
    #    losses[:,j] = medfilt(losses[:,j], kernel_size=11)
    #    #losses[:,j] = np.convolve(losses[:,j], np.ones((20,))/20, mode='same')

    # Labels
    if nvar == 6:
        labels = ['elbo', 'likelihood', 'KL', 'PDE', 'LR', 'UL']
    elif nvar == 4:
        labels = ['misfit', 'PDE', 'KL', 'Smoothness']
    elif nvar == 3:
        labels = ['PDE', 'KL', 'Smoothness']
    elif nvar == 2:
        labels = ['misfit', 'PDE']
    elif nvar == 1:
        labels = ['-1*likelihood']
    else:
        labels = [None] * nvar

    #fig, axes = plt.subplots(nrows=nvar, figsize=(7, 11)) # for external monitor
    fig, axes = plt.subplots(nrows=nvar, figsize=(10, 8)) # for laptop

    if nvar == 1:
        axes = [axes]

    for i in range(nvar):
        axes[i].plot(epochs, losses[:,i], alpha=0.7)
        axes[i].plot(epochs, losses[:,i+nvar], alpha=0.7)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, ls=':')

        #if nvar == 4 and i == 2:
        #    axes[i].set_ylim(0.0, 5)

    fig.set_tight_layout(True)
    plt.show()


def read_log(filename):
    with open(filename, 'r') as fid:
        epochs = []
        losses = []
        for input_line in fid:
            if not input_line.startswith('INFO:root'):
                continue
            fields = input_line.strip().split(':')
            dat = fields[-1].split()
            epoch = int(dat[0])
            loss = [float(x) for x in dat[1:]]
            epochs.append(epoch)
            losses.append(loss)

    return np.array(epochs), np.array(losses)


if __name__ == '__main__':
    args = parse_command_line()
    main(args)

# end of file
