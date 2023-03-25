"""
A collection of generic deep neural network functions.
"""

import numpy as np
from numpy.typing import NDArray


def conv2d(img: NDArray, weights: NDArray, biases: NDArray) -> NDArray:
    """
    Perform 2D convolution on the provided image using the provided kernel
    weights and biases.
    
    Parameters
    ==========
    img : array (in_channels, img_height, img_width) or (num_batches, in_channels, img_height, img_width)
        The input image, or a batch of images to process simultaneously.
    weights : array (out_channels, in_channels, kernel_height, kernel_width)
        The kernel to apply to the input.
    biases : array (out_channels, )
        The biases to add to each output channel.
    
    Returns
    =======
    array (out_channels, out_height, out_width) or (num_batches, out_channels, out_height, out_width)
        The input image convolved using the kernel and biases provided. Note
        that the output image size will be smaller than the input as no padding
        is performed during convolution.
        
        The num_batches dimension will be present iff it it was included in the
        input img.
    """
    # Internally always treat the input as batched
    batched = True
    if len(img.shape) == 3:
        img = img.reshape(1, *img.shape)
        batched = False
    
    
    num_batches = img.shape[0]
    out_channels, in_channels, kernel_height, kernel_width = weights.shape
    
    # Produce a sliding window over the input image to which the kernel will be
    # applied.
    windows = np.lib.stride_tricks.sliding_window_view(
        img,  # (num_batches, in_channels, img_height, img_width)
        (num_batches, in_channels, kernel_height, kernel_width)
    )  # (1, 1, out_height, out_width, num_batches, in_channels, kernel_height, kernel_width)
    
    # Remove extra dimensions created by sliding_window_view for the
    # two non-windowed dimensions:
    #
    # (out_height, out_width, num_batches, in_channels, kernel_height, kernel_width)
    windows = windows.squeeze(0).squeeze(0)
    
    # Apply the convolution kernel
    out = np.einsum(
        "hwbikl,oikl->bohw",
        # (out_height:h, out_width:w, num_batches:b, in_channels:i, kernel_height:k, kernel_width:l)
        windows,
        # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
        weights,
        optimize=True,  # Free 10x speedup :)
    )  # (num_batches, out_channels, out_height, out_width)
    
    # Add biases
    out += biases.reshape(out_channels, 1, 1)  # Reshaped for broadcasting
    
    if batched:
        return out
    else:
        return out[0]


def prelu(x: NDArray, parameters: NDArray, axis: int) -> NDArray:
    """
    Implements Parameterised Rectified Linear Unit (PReLU).
    
    Values in ``x`` are mapped to themselves if greater than zero or scaled by
    the value in ``parameters`` if less than zero.
    
    Parameters
    ==========
    x : array (any shape)
        The input array.
    parameters : array (1-dimensional array)
        The scaling factors used for negative values in x. The scaling value in
        ``parameters`` used for a given value in ``x`` is based on its location
        along the axis identified by the ``axis`` parameter.
        
        This array must have the same length as the chosen axis in ``x``.
    axis : int
        The axis index in ``x`` to use to select the scaling parameter in
        ``parameters``.
    
    Returns
    =======
    array (same shape as ``x``)
    """
    # Reshape to broadcast to the selected axis
    parameters = parameters.reshape(*(-1 if i == axis else 1 for i in range(x.ndim)))
    
    # NB: This method is kind of slow compared to PyTorch's implementation
    # which (I imagine) has a nicely vectorised compare-and-scale routine going
    # on.
    #
    # Depending on how your measure it, this code runs between 2 and 10 times
    # slower(!)
    return (
        x +
        ((x < 0) * x * (parameters - 1))
    )


def max_pool_2d(x: NDArray, block_height: int, block_width: int) -> NDArray:
    """
    Apply 2D 'max pooling' in which two dimensions of the input are reduced in
    size by breaking them into blocks and taking the maximum value in each
    block.
    
    If the input is an array::
    
        +-----------+
        |           |
        |           |
        |           |
        +-----------+
    
    It is broken up into blocks of size block_height and block_width::
    
        +---+---+---+
        |   |   |   |
        +---+---+---+
        |   |   |   |
        +---+---+---+
    
    The maximum value within each block is then computed and returned as a new
    (smaller) 2D array.
    
    Parameters
    ==========
    x : array (..., in_height, in_width)
        The input array. The final two dimensions are used as 2D coordinates.
        They must be exact multiples of block_height and block_width.
    block_height : int
    block_width : int
        The size of the blocks the input will be divided into.
    
    Returns
    =======
    array (..., in_height / block_height, in_width / block_width)
        The maxima for each input block.
    """
    # Get a windowed view of 'x's final two dimensions
    windowed = np.lib.stride_tricks.sliding_window_view(
        x,
        x.shape[:-2] + (block_height, block_width),
    )
    # Drop extra windowing dimensions corresponding to the non-coordinate axes
    for _ in range(x.ndim - 2):
        windowed = windowed.squeeze(0)
    
    # windowed.shape == (wh, ww, ..., block_height, block_width)
    
    # Since sliding_window_view returns all possible overlapping windows, we
    # want to pull out just the non-overlapping set
    #
    # (out_height, out_width, ..., block_height, block_width)
    windowed = windowed[::block_height, ::block_width, ...]
    
    # We now perform max pooling
    out = windowed.max(axis=(-2, -1))  # (out_height, out_width, ...)
    
    # And finally move the window dimensions to the end
    out = np.moveaxis(out, (0, 1), (-2, -1))  # (..., out_height, out_width)
    
    return out


def softmax(x: NDArray, axis: int = -1) -> NDArray:
    """
    A numerically stable implementation of the soft(arg)max function.
    """
    # For reasons of numerical stability, offset all values such that we don't
    # inadvertently produce an overflow after exponentiation.
    #
    # NB: The subtraction has no effect on the outcome otherwise, e.g. because:
    #
    #     e^a / (e^a + e^b) = e^(a-x)      / (e^(a-x)      + e^(b-x))
    #                       = (e^a * e^-x) / ((e^a * e^-x) + (e^b * e^-x))
    #                       = (e^a * e^-x) / ((e^a         + e^b          ) * e^-x)
    #                       =  e^a         / ( e^a         + e^b)
    #
    # NB: We perform the subtraction locally within each axis to avoid excessively scaling down
    # unrelated values which also introduces numerical stability issues but in the opposite
    # direction.
    x = x - np.max(x, axis=axis, keepdims=True)

    x = np.exp(x)
    return x / np.sum(x, axis=axis, keepdims=True)
