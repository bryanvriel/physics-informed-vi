#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from utilities import *
from models import *
import matplotlib.pyplot as plt
from functools import partial
import argparse

usage = """
Usage: pretrain_solution.py run.cfg
"""

def main(pars):

    print('Restore:', pars.pretrain.restore)

    # Load data and normalizers
    data, norms = make_pretrain_data()

    # Create mean network
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet)

    # Define reconstruction loss for surface variables
    @tf.function
    def loss_function(batch):
        """                                  
        Loss function only for predicting observations. Use likelihood based sampling loss.
        """
        # Predict at data coordinates (for computing MSE loss)
        up, vp, hp, bp = mean_net(batch.x, batch.y, inverse_norm=False)

        # Compute negative log-likelihood
        nll = tf.reduce_mean(
            (tf.square(up - batch.u) / tf.square(batch.u_err)) +
            (tf.square(vp - batch.v) / tf.square(batch.v_err)) +
            (tf.square(hp - batch.h) / tf.square(batch.h_err)) +
            (tf.square(bp - batch.b) / tf.square(batch.b_err))
        )
        
        return [nll,]

    # JIT and evaluate loss function in order to get access to trainable variables
    batch = data.test_batch()
    lv = loss_function(batch)
    trainable_variables = mean_net.trainable_variables
    for v in trainable_variables:
        print(v.name, v.shape)

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
