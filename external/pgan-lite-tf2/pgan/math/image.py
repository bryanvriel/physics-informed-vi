#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def build_sobel_filters():
    """
    Construct arrays to be used for 2D convolution of Sobel filters for both
    horizontal and vertical gradients. Tensorflow expects filters to of shape:

        [filter_height, filter_width, in_channels, out_channels]

    """
    # For horizontal gradients
    H = np.zeros((3, 3, 1, 1))
    H[:,:,0,0] = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # For vertical gradients
    V = np.zeros((3, 3, 1, 1))
    V[:,:,0,0] = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Convert to tensorflow tensors
    H = tf.convert_to_tensor(H, dtype=tf.float32)
    V = tf.convert_to_tensor(V, dtype=tf.float32)

    return H, V

def build_correction_tensors(Ny, Nx):
    """
    Construct correction tensors that are applied to images AFTER convolution with the
    Sobel filters. These tensors essentially perform finite differences on the boundaries
    of the spatial domain to improve accuracy.
    """
    # For horizontal gradients
    M_h = np.eye(Nx)
    M_h[0,0] = -4
    M_h[1,0] = 1
    M_h[-1,-1] = -4
    M_h[-2,-1] = 1

    # For vertical gradients
    M_v = np.eye(Ny)
    M_v[0,0] = -4
    M_v[0,1] = 1
    M_v[-1,-1] = -4
    M_v[-1,-2] = 1

    # Convert to tensorflow tensors
    M_h = tf.convert_to_tensor(M_h, dtype=tf.float32)
    M_v = tf.convert_to_tensor(M_v, dtype=tf.float32)

    return M_h, M_v

def compute_boundary_loss(W, scale=1.0):
    """
    Compute periodic boundary loss.
    """
    # Extract the boundary slices
    top = W[:,0,:,0]
    bot = W[:,-1,:,0]
    left = W[:,:,0,0]
    right = W[:,:,-1,0]

    # Compute periodic misfits
    return scale * (tf.reduce_mean(tf.square(top - bot)) +
                    tf.reduce_mean(tf.square(left - right)))

def compute_grad(w, S, M, mode='horizontal'):
    """
    Apply three-step process of:
        1. Padding
        2. Convolution
        3. Correction

    Returns UNITLESS gradients.
    """
    # Pad
    out = pad(w, 1, mode='reflect')

    # Convolution
    out = tf.nn.conv2d(out, filter=S, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')

    # Correction
    if mode == 'horizontal':
        out = tf.einsum('ijkl,km->ijml', out, M)
    elif mode == 'vertical':
        out = tf.einsum('mj,ijkl->imkl', M, out)
    else:
        raise ValueError('grad mode must be horizontal or vertical')

    # Done
    return out

def pad(x_input, p, mode='reflect'):
    """
    Convenience function for padding a rank 4 tensor. Assumes NHWC format.
    """
    return tf.pad(x_input, [[0, 0], [p, p], [p, p], [0, 0]], mode=mode)

def image_gradient(w, d, mode='vertical'):
    """
    Implements finite difference in space. Equivalent to np.gradient.
    """
    # Vertical gradients
    if mode == 'vertical':

        # Interior
        grad_interior = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2.0 * d)

        # First edge forward difference
        grad_first = (w[:, 1, :, :] - w[:, 0, :, :]) / d
        grad_first = tf.expand_dims(grad_first, axis=1)

        # Last edge backward difference
        grad_last = (w[:, -1, :, :] - w[:, -2, :, :]) / d
        grad_last = tf.expand_dims(grad_last, axis=1)

        # Concatenate
        grad = tf.concat(values=[grad_first, grad_interior, grad_last], axis=1)

    # Horizontal gradients
    elif mode == 'horizontal':

        # Interior
        grad_interior = (w[:, :, 2:, :] - w[:, :, :-2, :]) / (2.0 * d)

        # First edge forward difference
        grad_first = (w[:, :, 1, :] - w[:, :, 0, :]) / d
        grad_first = tf.expand_dims(grad_first, axis=2)

        # Last edge backward difference
        grad_last = (w[:, :, -1, :] - w[:, :, -2, :]) / d
        grad_last = tf.expand_dims(grad_last, axis=2)

        # Concatenate
        grad = tf.concat(values=[grad_first, grad_interior, grad_last], axis=2)

    else:
        raise ValueError('Unsupported gradient mode.')

    return grad

def image_gradient_2nd_order(w, d, mode='vertical'):
    """
    Implements 2nd-order finite difference in space. Equivalent to np.gradient.
    """
    # Vertical gradients
    if mode == 'vertical':

        # Interior
        grad_interior = (w[:, 2:, :, :] - 2.0 * w[:, 1:-1, :, :] + w[:, :-2, :, :]) / (2.0 * d)

        # First edge forward difference
        grad_first = (w[:, 2, :, :] - 2.0 * w[:, 1, :, :] - w[:, 0, :, :]) / d
        grad_first = tf.expand_dims(grad_first, axis=1)

        # Last edge backward difference
        grad_last = (w[:, -1, :, :] - 2.0 * w[:, -2, :, :] + w[:, -3, :, :]) / d
        grad_last = tf.expand_dims(grad_last, axis=1)

        # Concatenate
        grad = tf.concat(values=[grad_first, grad_interior, grad_last], axis=1)

    # Horizontal gradients
    elif mode == 'horizontal':

        # Interior
        grad_interior = (w[:, :, 2:, :] - 2.0 * w[:, :, 1:-1, :] + w[:, :, :-2, :]) / (2.0 * d)

        # First edge forward difference
        grad_first = (w[:, :, 2, :] - 2.0 * w[:, :, 1, :] - w[:, :, 0, :]) / d
        grad_first = tf.expand_dims(grad_first, axis=1)

        # Last edge backward difference
        grad_last = (w[:, :, -1, :] - 2.0 * w[:, :, -2, :] + w[:, :, -3, :]) / d
        grad_last = tf.expand_dims(grad_last, axis=1)

        # Concatenate
        grad = tf.concat(values=[grad_first, grad_interior, grad_last], axis=2)

    else:
        raise ValueError('Unsupported gradient mode.')

    return grad

# end of file
