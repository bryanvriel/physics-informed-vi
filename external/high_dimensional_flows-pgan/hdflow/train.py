#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging

def train(variables, loss_function, dataset, optimizer, ckpt_manager,
          n_epochs=100, clip=None, logfile='log_train', ckpt_skip=10, **kwargs):
    """
    Custom training loop to fetch batches from datasets with different sizes
    and perform updates on variables.

    Parameters
    ----------
    variables: list
        List of trainable variables.
    loss_function:
        Function that returns list of losses.
    dataset: tf.data.Dataset
        Training dataset.
    test_dataset: tf.data.Dataset
        Testing dataset.
    optimizer: tf.keras.optimizers.Optimizer
        Optimizer.
    ckpt_manager: tf.train.CheckpointManager
        Checkpoint manager for saving weights.
    n_epochs: int, optional
        Number of epochs. Default: 100.
    clip: int or float, optional
        Global norm value to clip gradients. Default: None.
    logfile: str, optional
        Output file for logging training statistics. Default: 'log_train'.
    ckpt_skip: int, optional
        Save weights every `ckpt_skip` epochs. Default: 10.
    **kwargs:
        kwargs to pass to loss_function.

    Returns
    -------
    None
    """
    # Reset logging file
    logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)

    # Evaluate loss function on batch in order to get number of losses
    batch = dataset.train_batch()
    losses = loss_function(batch, **kwargs)
    n_loss = len(losses)
    dataset.reset_training()

    # Loop over epochs
    train_epoch = np.zeros((n_epochs, n_loss))
    test_epoch = np.zeros((n_epochs, n_loss))
    for epoch in tqdm(range(n_epochs)):

        # Loop over batches and update variables, keeping batch of training stats
        train = np.zeros((dataset.n_batches, n_loss))
        for cnt in range(dataset.n_batches):
            batch = dataset.train_batch()
            losses = update(variables, loss_function, optimizer, batch)
            train[cnt, :] = [value.numpy() for value in losses]

        # Compute mean train loss
        train = np.mean(train, axis=0).tolist()

        # Evalute losses on test batch
        try:
            batch = dataset.test_batch()
            losses = loss_function(batch, **kwargs)
            test = [value.numpy() for value in losses]
        except ValueError:
            test = train

        # Reshuffle
        dataset.reset_training()

        # Store in epoch arrays
        train_epoch[epoch, :] = train
        test_epoch[epoch, :] = test

        # Write stats to logfile
        out = '%d ' + '%15.10f ' * 2 * n_loss
        logging.info(out % tuple([epoch] + train + test))

        # Periodically save checkpoint
        if epoch > 0 and epoch % ckpt_skip == 0:
            ckpt_manager.save()

    # Save final checkpoint
    ckpt_manager.save()

    # Return stats
    return train_epoch, test_epoch

@tf.function
def update(variables, loss_function, optimizer, batch, clip=None, **kwargs):
    """

    Parameters
    ----------
    variables: list
        List of trainable variables.
    loss_function:
        Function that returns list of losses.
    optimizer: tf.keras.optimizers.Optimizer
        Optimizer.
    batch:
        tf.data.Dataset batch passed to loss function.
    clip: int, float, or None
        Global norm value to clip gradients. Default: None.
    **kwargs:
        Additional kwargs passed to loss function.

    Returns
    -------
    losses: list
        List of scalar losses.
    """
    # Compute gradient of total loss
    with tf.GradientTape() as tape:
        losses = loss_function(batch, **kwargs)
        total_loss = sum(losses)
    grads = tape.gradient(total_loss, variables)

    # Clip gradients
    if clip is not None:
        grads, _ = tf.clip_by_global_norm(grads, clip)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, variables))

    # Return all losses
    return losses

def create_checkpoint_manager(checkdir, restore=False, max_to_keep=1, **kwargs):
    """
    Convenience function for creating a checkpoint manager for loading and
    saving model weights during training.

    Parameters
    ----------
    checkdir: str
        Checkpoint directory to save to and load from.
    restore: bool
        Restore previously saved weights. Default: False.
    max_to_keep: int
        Maximum number of checkpoints to save in checkdir. Default: 1.
    **kwargs:
        kwargs passed to tf.train.Checkpoint.

    Returns
    -------
    ckpt_manager: tf.train.CheckpointManager
        The checkpoint manager.
    """
    ckpt = tf.train.Checkpoint(**kwargs)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkdir, max_to_keep)
    if restore:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    return ckpt_manager


# end of file
