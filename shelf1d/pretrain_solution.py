#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import h5py
import sys
import os

from models import *
from utilities import *

usage = """
Usage: train_joint.py run.cfg
"""

def main(pars):

    print('Restore:', pars.pretrain.restore)
    print('Resnet:', pars.pretrain.resnet)

    # Make data objects
    data_surface, data_physics, norms = make_data(
        batch_size=pars.pretrain.batch_size,
        physics_batch_size=pars.train.batch_size
    )
    # Combine into a collection
    data = DataCollection(data_surface, data_physics)

    # Create mean network
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet)

    # Define reconstruction loss for surface variables
    @tf.function
    def loss_function(batch):

        # Unpack batch
        batch_surface, batch_physics = batch
        Nb = tf.cast(tf.shape(batch_physics.x)[0], dtype=DTYPE)

        # Predict normalized outputs
        up, hp = mean_net(batch_surface.x, inverse_norm=False)
        # Negative log-likelihood
        data_nll = tf.reduce_mean(
            (tf.square(up - batch_surface.u) / tf.square(batch_surface.u_err)) +
            (tf.square(hp - batch_surface.h) / tf.square(batch_surface.h_err))
        )

        # Predict again at physics coordinates
        up, hp = mean_net(batch_physics.x, inverse_norm=True)

        # First-order gradient smoothness loss
        u_x = tf.gradients(up, batch_physics.x)
        h_x = tf.gradients(hp, batch_physics.x)
        smooth = 20 * tf.reduce_mean(tf.square(u_x) + tf.square(h_x))
        
        return [data_nll, smooth]

    # JIT and evaluate loss function in order to get access to trainable variables
    batch = data.train_batch()
    lv = loss_function(batch)
    trainable_variables = mean_net.trainable_variables
    print('\nTrainable variables:')
    for v in trainable_variables:
        print(v.name, v.shape)
    print('')

    # Create optimizer
    optimizer = create_optimizer(pars.train.optimizer, learning_rate=pars.pretrain.lr)

    # Create checkpoint manager for saving model parameters
    CHECKDIR = pars.pretrain.checkdir
    ckpt_manager = create_checkpoint_manager(
        CHECKDIR,
        mean_net=mean_net,
        optimizer=optimizer,
        restore=pars.pretrain.restore
    )

    # Run optimization
    train_vals, test_vals = train(
        trainable_variables, loss_function, data, optimizer, ckpt_manager,
        n_epochs=pars.pretrain.n_epochs, clip=5, logfile=pars.pretrain.logfile
    )


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print(usage)
        sys.exit()
    pars = ParameterClass(args[0])
    main(pars)

# end of file
