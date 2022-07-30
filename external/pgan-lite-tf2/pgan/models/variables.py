#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class MultiVariable(dict):
    """
    Class for representing multi-component input and output variables. Simple extension
    of Python dict with dot operator for accessing attributes and sum and concatenation
    methods. Note that since Python 3.7, dicts are ordered by insertion.
    """

    def __init__(self, *args, **kwargs):
        """
        Create N-d variables by dictionary unpacking or extracting columns from
        a column-stacked tensor.
        """
        # Special case: extract columns from tensor provided in positional argument
        if len(args) == 1 and isinstance(args[0], (tf.Tensor, np.ndarray)):

            # Get shape of tensor
            dat = args[0]
            N_batch, N_col = tf.shape(dat)

            # Loop over variables to get total number of dimensions
            ndims = 0
            for varname, ndim in kwargs.items():
                ndims += ndim
            assert ndims == N_col

            # Set variables
            for cnt, (varname, ndim) in enumerate(kwargs.items()):
                super().__setitem__(varname, tf.reshape(dat[:, cnt], (-1, ndim)))

        # Otherwise, init the parent dict
        else:
            super().__init__(*args, **kwargs)
        
    def concat(self, var_list=None):
        """
        Concatenates individual variables along the last dimension.
        """
        # List all variables
        if var_list is None:
            values = self.values()

        # Or specific variables
        else:
            values = [self.vars[name] for name in var_list]

        # Concatenate and return
        return tf.concat(values=values, axis=-1)

    def names(self):
        """
        Return the variable names.
        """
        return list(self.keys())

    def sum(self):
        """
        Returns sum over all variables. Tensorflow will broacast dimensions when possible.
        """
        return sum([value for value in self.vars.values()])

    def __getattr__(self, attr):
        """
        Return specific variable value using dot(.) operator.
        """
        return self.get(attr)

    def __setattr__(self, key, value):
        """
        Set specific variable value using dot(.) operator.
        """
        self.__setitem__(key, value)


# end of file
