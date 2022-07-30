#-*- coding: utf-8 -*-

import tensorflow as tf
from collections import OrderedDict
import shutil
import os


class Summary:
    """
    Convenience class for writing tensorflow summaries to be displayed with Tensorboard.
    Writes train and test loss values to the same plot.
    """

    def __init__(self, sess, losses, outdir='summaries'):
        """
        Creates placeholder and tf.summary.scalar objects for each item in losses, which
        may be one of the following:

        a) a pgan.models.MultiVariable instance
        b) list of pgan.models.MultiVariable instances
        
        """
        # Clean or ensure output directory exists
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        else:
            shutil.rmtree(outdir)
            os.makedirs(outdir)

        # Unpack the loss names and nodes
        if isinstance(losses, pgan.models.MultiVariable):
            self.loss_names = losses.names()
            self.loss_nodes = losses.values()
        elif isinstance(losses, list):
            self.loss_names = []
            self.loss_nodes = []
            for loss in losses:
                self.loss_names.extend(loss.names())
                self.loss_nodes.extend(loss.values())
        else:
            raise ValueError('Incompatible loss type. Must be MultiVariable or list.')

        # Initialize empty (ordered) dictionaries
        self.placeholders = OrderedDict()
        self.train_writers = OrderedDict()
        self.test_writers = OrderedDict()
        self.summaries = OrderedDict()

        # Create plaeholders and summaries for each loss item
        summaries = []
        for loss_name in self.loss_names:
            
            # Create placeholder for loss value
            ph_name = '%s_value' % loss_name
            self.placeholders[loss_name] = tf.placeholder(tf.float32, name=ph_name)

            # Create tensorflow summary from loss value
            self.summaries[loss_name] = tf.summary.scalar(loss_name, self.placeholders[loss_name])

            # Create writers for train and test summaries
            loss_outdir = os.path.join(outdir, loss_name)
            self.train_writers[loss_name] = tf.summary.FileWriter(
                os.path.join(loss_outdir, 'train'), sess.graph
            )
            self.test_writers[loss_name] = tf.summary.FileWriter(
                os.path.join(loss_outdir, 'test'), sess.graph
            )

    def write_summary(self, sess, feed_dict, iternum, loss_values=None, stype='train'):

        # Evaluate loss nodes with feed dict if none are provided
        if loss_values is None:
            loss_values = sess.run(self.loss_nodes, feed_dict=feed_dict)
        # Otherwise, make sure we've passed in enough values
        else:
            assert len(loss_values) == len(self.loss_names), \
                'Incompatible number of loss values for summary.'

        # Construct feed dictionary for loss placeholders
        feed_dict = {}
        for cnt, loss_name in enumerate(self.loss_names):
            feed_dict[self.placeholders[loss_name]] = loss_values[cnt]

        # Point to right summary writers
        if stype == 'train':
            writers = self.train_writers
        elif stype == 'test':
            writers = self.test_writers
        else:
            raise ValueError('Invalid stype for specifying summary writer.')

        # Add summaries and flush
        for loss_name in self.loss_names:
            summ = sess.run(self.summaries[loss_name], feed_dict)
            writers[loss_name].add_summary(summ, iternum)
            writers[loss_name].flush()

        return loss_values

    @property
    def size(self):
        """
        Read-only property returning the number of loss nodes.
        """
        return len(self.loss_nodes)

    @size.setter
    def size(self, value):
        raise ValueError('Cannot set the summary size explicitly.')     


# end of file
