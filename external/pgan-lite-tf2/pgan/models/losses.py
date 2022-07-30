#-*- coding: utf-8 -*-

import tensorflow as tf
from .variables import MultiVariable

def mse(obs_var, pred_var, scale=tf.constant(1.0), name='mse'):
    """
    Computes MSE loss between observation and prediction MultiVariable instances.
    """
    # Check the variables have the same variable names
    assert set(obs_var.names()) == set(pred_var.names())

    # Iterate over variables to populate loss variable
    loss = MultiVariable()
    for varname in obs_var.names():
        key = '%s_%s' % (name, varname)
        loss[key] = scale * tf.reduce_mean(tf.square(obs_var[varname] - pred_var[varname]))

    # Done
    return loss


# end of file
