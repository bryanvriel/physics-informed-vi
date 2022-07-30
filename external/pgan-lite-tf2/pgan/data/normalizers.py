#-*- coding: utf-8 -*-

import numpy as np
from pgan.models import MultiVariable

class Normalizer:
    """
    Simple convenience class that performs transformations to/from normalized values.
    Here, we use the norm range [-1, 1] for pos=False or [0, 1] for pos=True.
    """

    def __init__(self, xmin, xmax, pos=False, log=False):
        self.xmin = xmin
        self.xmax = xmax
        self.denom = xmax - xmin
        self.pos = pos
        self.log = log
        self.log_eps = 0.05

    def __call__(self, x):
        """
        Alias for Normalizer.forward()
        """
        return self.forward(x)

    def forward(self, x):
        """
        Normalize data.
        """
        if self.pos:
            return (x - self.xmin) / self.denom
        elif self.log:
            xn = (x - self.xmin + self.log_eps) / self.denom
            return np.log(xn)
        else:
            return 2.0 * (x - self.xmin) / self.denom - 1.0

    def forward_scale(self, scale):
        """
        Normalize a scale factor (e.g., a standard deviation).
        """
        if self.pos:
            return scale / self.denom
        elif self.log:
            raise NotImplementedError
        else:
            return 2 * scale / self.denom

    def inverse(self, xn):
        """
        Un-normalize data.
        """
        if self.pos:
            return self.denom * xn + self.xmin
        elif self.log:
            return self.denom * np.exp(xn) + self.xmin - self.log_eps
        else:
            return 0.5 * self.denom * (xn + 1.0) + self.xmin

    def inverse_scale(self, scale, *args):
        """
        Un-normalize a scale factor (e.g., a standard deviation).
        """
        if self.pos:
            return self.denom * scale
        elif self.log:
            return (args[0] - self.xmin + self.log_eps) * scale
        else:
            return 0.5 * self.denom * scale


class MultiNormalizer:
    """
    Encapsulates multiple Normalizer objects hashed by name.
    """

    def __init__(self, **kwargs):
        self.normalizers = {}
        for name, norm in kwargs.items():
            assert isinstance(norm, Normalizer), 'Must pass in Normalizer as value'
            self.normalizers[name] = norm

    def __call__(self, multi_var):
        """
        Alias for MultiNormalizer.forward().
        """
        return self.forward(multi_var)

    def forward(self, multi_var):
        """
        Performs normalization (forward pass) of MultiVariable instance. Returns a
        new MultiVariable instance.
        """
        # Initialize output variable
        out = MultiVariable()

        # Iterate over variable names
        for varname, normalizer in self.normalizers.items():
            out[varname] = normalizer(multi_var[varname])

        # Done
        return out

    def inverse(self, multi_var):
        """
        Performs inverse normalization (un-normalize) of MultiVariable instance. Returns a
        new MultiVariable instance.
        """
        # Initialize output variable
        out = MultiVariable()

        # Iterate over variable names
        for varname, normalizer in self.normalizers.items():
            out[varname] = normalizer.inverse(multi_var[varname])

        # Done
        return out


def compute_bounds(x, n_sigma=1.0, method='normal'):
    """
    Convenience method for computing reasonable normalization bounds for a given
    data array. Uses either mean +/- n_sigma*stddev or [minval, maxval].
    """
    if method == 'normal':
        mean = np.mean(x)
        std = np.std(x)
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        return [lower, upper]
    elif method == 'minmax':
        maxval = np.nanmax(x)
        minval = np.nanmin(x)
        return [minval, maxval]
    else:
        raise ValueError('Unsupported bounds determination method')


# end of file
