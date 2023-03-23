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


