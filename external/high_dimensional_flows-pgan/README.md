# High dimensional variational inference with normalizing flows

This package contains TensorFlow implementations for performing variational inference on high-dimensional inverse problems. High-dimensional, spatial fields of physical parameters are approximated with a neural network that inputs spatial coordinates. Thus, we can formulate a base, mean-field surrogate distribution that is conditioned on those spatial coordinates.

See the `examples` directory for different use cases.
