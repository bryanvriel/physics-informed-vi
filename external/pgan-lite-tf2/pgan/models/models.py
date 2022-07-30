#-*- coding: utf-8 -*-

import tensorflow as tf
from pgan.logging import Summary
from tqdm import tqdm
import logging
import os

class Model(tf.keras.Model):
    """
    Abstract class for all trainable models that encapsulate multiple sub-models.
    """

    def __init__(self, *args, name='model', **kwargs):
        """
        Initialize keras Model class and necessary variables.
        """
        # The parent class
        super().__init__(name=name)
        # Initialize dictionary of submodels
        self.submodels = {}
        # Initialize dictionary of savers
        self.savers = {}
        return

    def build(self, graph=None, inter_op_cores=None, intra_op_threads=None,
              summary_dir='summaries'):
        """
        Initialize session AFTER all graphs have been built.
        """
        # If cores/threads info is passed, make a custom configuration
        if inter_op_cores is not None and intra_op_threads is not None:
            config = tf.ConfigProto()
            config.intra_op_parallelism_threads = intra_op_threads
            config.inter_op_parallelism_threads = inter_op_cores
        else:
            config = None

        # Initialize session
        self.sess = tf.Session(graph=graph, config=config)

        # Create summary writer
        self.summary = Summary(self.sess, outdir=summary_dir)

        # Initialize global variables
        self.sess.run(tf.global_variables_initializer())
        
        # Create savers for submodel variables
        self.savers = {}
        for name, model in self.submodels.items():
            self.savers[name] = tf.train.Saver(var_list=model.trainable_variables)

        return

    def print_variables(self):
        """
        Print layer names. Must be called after self.build()
        """
        for name, model in self.submodels.items():
            print('Model:', name)
            for var in model.trainable_variables:
                print('   ', var.name)
        return

    def save(self, outdir='checkpoints', model=None):
        """
        Save model weights to file.
        """
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        if model is None:
            for name, saver in self.savers.items():
                saver.save(self.sess, os.path.join(outdir, '%s.ckpt' % name))
        else:
            self.savers[model].save(self.sess, os.path.join(outdir, '%s.ckpt' % model))

        return

    def load(self, indir='checkpoints', model=None):
        """
        Load model weights from file.
        """
        if model is None:
            for name, saver in self.savers.items():
                saver.restore(self.sess, os.path.join(indir, '%s.ckpt' % name))
                print('Restoring', name)
        else:
            self.savers[model].restore(self.sess, os.path.join(indir, '%s.ckpt' % model))

        return

    def train(self,
              data,
              n_iterations=100000,
              learning_rate=0.0001,
              verbose=True):
        """
        Run training for simple architectures and single training objective.
        """
        # Compute time scale for exponential cooling of learning rate if tuple provided
        if isinstance(learning_rate, tuple):
            initial_learning_rate, final_learning_rate = learning_rate
            lr_tau = -n_iterations / np.log(final_learning_rate / initial_learning_rate)
            print('Learning rate tau:', lr_tau)
        else:
            print('Using constant learning rate:', learning_rate)
            lr_tau = None

        # Training iterations
        for iternum in tqdm(range(n_iterations)):

            # Compute learning rate
            if lr_tau is not None:
                lr_val = initial_learning_rate * np.exp(-iternum / lr_tau)
            else:
                lr_val = learning_rate

            # Get batch of training data
            batch = data.train_batch()

            # Construct feed dictionary
            feed_dict = self.constructFeedDict(batch, None, lr_val=lr_val)

            # Run weight updates and compute training loss
            values = self.sess.run([self.train_op] + self._losses, feed_dict=feed_dict)
            # For some reason, tensorflow sticks update return value at the end
            train = values[:-1]

            # Run losses periodically for test data
            if iternum % 200 == 0:
                test_feed_dict = self.constructFeedDict(data.test, None)
                test = self.sess.run(self._losses, feed_dict=test_feed_dict)

            # Log training performance
            if verbose:
                out = '%d ' + '%f ' * 2 * len(self._losses)
                logging.info(out % tuple([iternum] + train + test))

            if iternum % 5000 == 0 and iternum != 0:
                self.save(outdir='temp_checkpoints')

        return

    def traingan(self,
                 data_gan,
                 data_pde,
                 n_iterations=100000,
                 dskip=5,
                 learning_rate=0.0001,
                 verbose=True):
        """
        Run training for GAN architectures.

            n_epochs: int
                Number of epochs (each epoch loops over all batches)
            dskip: int
                Skip factor for discriminator training ops.
        """
        # Compute time scale for exponential cooling of learning rate if tuple provided
        if isinstance(learning_rate, tuple):
            initial_learning_rate, final_learning_rate = learning_rate
            lr_tau = -n_iterations / np.log(final_learning_rate / initial_learning_rate)
            print('Learning rate tau:', lr_tau)
        else:
            print('Using constant learning rate:', learning_rate)
            lr_tau = None

        # Training iterations
        for iternum in tqdm(range(n_iterations)):

            # Compute learning rate
            if lr_tau is not None:
                lr_val = initial_learning_rate * np.exp(-iternum / lr_tau)
            else:
                lr_val = learning_rate

            # Get batch of training data
            batch_gan = data_gan.train_batch()
            batch_pde = data_pde.train_batch()

            # Construct feed dictionary
            feed_dict = self.constructFeedDict(batch_gan, batch_pde, lr_val=lr_val)

            # Run weight updates and compute training loss
            values = self.sess.run([self.gen_train_op] + self._losses, feed_dict=feed_dict)
            # For some reason, tensorflow sticks update return value at the end
            train = values[:-1]

            # Periodically run update for discriminator
            if iternum % dskip == 0:
                self.sess.run(self.disc_train_op, feed_dict=feed_dict)

            # Run losses periodically for test data
            if iternum % 200 == 0:
                test_feed_dict = self.constructFeedDict(data_gan.test, data_pde.test)
                test = self.sess.run(self._losses, feed_dict=test_feed_dict)
            
            # Log training performance
            if verbose:
                out = '%d ' + '%f ' * 2 * len(self._losses)
                logging.info(out % tuple([iternum] + train + test))

            if iternum % 5000 == 0 and iternum != 0:
                self.save(outdir='temp_checkpoints')

        return

    def trainvae(self,
                 data,
                 data_pde,
                 n_iterations=100000,
                 learning_rate=0.0001,
                 verbose=True):
        """
        Run training for VAE/feedforward architectures. Data objects for different training
        objective functions.
        """
        # Compute time scale for exponential cooling of learning rate if tuple provided
        if isinstance(learning_rate, tuple):
            initial_learning_rate, final_learning_rate = learning_rate
            lr_tau = -n_iterations / np.log(final_learning_rate / initial_learning_rate)
            print('Learning rate tau:', lr_tau)
        else:
            print('Using constant learning rate:', learning_rate)
            lr_tau = None

        # Training iterations
        for iternum in tqdm(range(n_iterations)):

            # Compute learning rate
            if lr_tau is not None:
                lr_val = initial_learning_rate * np.exp(-iternum / lr_tau)
            else:
                lr_val = learning_rate

            # Get batch of training data
            batch = data.train_batch()
            batch_pde = data_pde.train_batch()

            # Construct feed dictionary
            feed_dict = self.constructFeedDict(batch, batch_pde, lr_val=lr_val)

            # Run weight updates and compute training loss
            values = self.sess.run([self.train_op] + self._losses, feed_dict=feed_dict)
            train = values[1:]

            # Run losses periodically for test data
            if iternum % 200 == 0:
                test_feed_dict = self.constructFeedDict(data.test, data_pde.test)
                test = self.sess.run(self._losses, feed_dict=test_feed_dict)

            # Log training performance
            if verbose:
                out = '%d ' + '%f ' * 2 * len(self._losses)
                logging.info(out % tuple([iternum] + train + test))

            if iternum % 5000 == 0 and iternum != 0:
                self.save(outdir='temp_checkpoints')

        return    

    def constructFeedDict(self, *args, **kwargs):
        raise NotImplementedError('Sublcasses must implement constructFeedDict.')


# end of file
