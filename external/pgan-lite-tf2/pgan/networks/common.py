#-*- coding: utf-8 -*-

import tensorflow as tf

class DenseNet(tf.keras.Model):
    """
    Generic feedforward neural network.
    """

    def __init__(self,
                 layer_sizes,
                 initializer='glorot_normal',
                 batch_norm=False,
                 dropout_rate=None,
                 dtype=tf.float32,
                 name='net'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create and store layers
        self.net_layers = []
        for count, size in enumerate(layer_sizes):
            # Layer names by depth count
            name = 'dense_%d' % count
            self.net_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=None,
                    kernel_initializer=initializer,
                    dtype=dtype,
                    name=name
                )
            )
        self.n_layers = len(self.net_layers)

        # Create batch norm layers
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm_layers = []
            for count in range(self.n_layers - 1):
                self.norm_layers.append(tf.keras.layers.BatchNormalization())

        # Create dropout layers
        if dropout_rate is not None:
            self.dropout = True
            self.dropout_layers = []
            for count in range(self.n_layers - 1):
                self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))
        else:
            self.dropout = False

        return

    def call(self, inputs, activation='tanh', actfun=None, training=False, activate_outputs=False):
        """
        Pass inputs through network and generate an output. All layers except the last
        layer will pass through an activation function.

        NOTE: Do not call this directly. Use instance __call__() functionality.
        """
        # Cache activation function
        if actfun is None and activation is not None:
            actfun = getattr(tf.nn, activation)

        # Pass through all layers, use activations in all but last layer
        out = inputs
        for cnt, layer in enumerate(self.net_layers):

            # Pass through weights
            out = layer(out)

            # If not output layer...
            if cnt != (self.n_layers - 1):

                # Apply optional batch normalization
                if self.batch_norm:
                    out = self.norm_layers[cnt](out, training=training)

                # Apply activation                
                out = actfun(out)

                # Apply dropout
                if self.dropout:
                    out = self.dropout_layers[cnt](out, training=training)

        # Pass outputs through activation function if requested
        if activate_outputs:
            out = actfun(out)

        # Done
        return out

# end of file
