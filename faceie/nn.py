"""
A collection of generic deep neural network functions.
"""

import numpy as np
from numpy.typing import NDArray

from math import ceil, floor


def linear(x: NDArray, weights: NDArray, biases: NDArray) -> NDArray:
    """
    A simple linear/dense layer: (x @ weights) + biases
    """
    return (x @ weights) + biases


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
        (num_batches, in_channels, kernel_height, kernel_width),
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
    return x + ((x < 0) * x * (parameters - 1))


def max_pool_2d(
    x: NDArray,
    kernel: int | tuple[int, int],
    stride: int | tuple[int, int],
    ceil_mode: bool = True,
    out: NDArray | None = None,
) -> NDArray:
    """
    Apply 2D 'max pooling' convolution filter in which the input is convolved
    using the 'max' function acting over kernel-sized regions.

    .. warning::

        This function is significantly slower than similar functions in (e.g.)
        PyTorch due to numpy's apparent lack of SIMD or multicore support in
        its 'amax' function.

    Parameters
    ==========
    x : array (..., in_height, in_width)
        The input array. The final two dimensions are used as 2D coordinates.
        They must be exact multiples of block_height and block_width.
    kernel : int or (int, int)
        The kernel height and width.
    stride : int or (int, int)
        The step size.
    ceil_mode : bool
        If True, add -inf padding to the right and bottom edges of the input
        when necessary to ensure that all input values are represented in the
        output.

        For some combinations of input size, kernel and stride, it is possible
        for some input values to be omitted from processing. The illustration
        below shows the regions of an input processed by a kernel::

            +---+---+---+-+   ceil_mode == False
            |   |   |   |X|
            +---+---+---+X|   X = values not processed by any kernel
            |   |   |   |X|
            +---+---+---+X|
            +XXXXXXXXXXXXX+

        Here, because the kernel and stride do not exactly divide the input,
        some inputs at the bottom and right sides are not processed by any
        kernels and are effectively ignored. This is the behaviour when
        ``ceil_mode`` is False.

        When ``ceil_mode`` is True, the input is effectively padded with -inf
        to enable an extra kernel to fit, capturing the edge-most values::

            +---+---+---+---+   ceil_mode == True
            |   |   |   | PP|
            +---+---+---+---+   P = padding '-inf' values added to input
            |   |   |   | PP|
            +---+---+---+---+
            |   |   |   | PP|
            +PPP+PPP+PPP+PPP+

        The result is that the output size grows by one and all input values
        are processed by at least one kernel. (Unless of course the stride is
        larger than the kernel!)

        .. note::

            The case where the stride is larger than the kernel is considered
            degenerate in that inputs will be ignored between kernels which fit
            in the input without padding. In this case, ceil_mode will continue
            to expand the output

    out: NDArray or None
        If given, an array into which to write the output (see return value for
        expected size). Otherwise, a new array will be allocated.

    Returns
    =======
    array (..., out_height, out_width)
        The output of the convolution.

        The output dimensions are:

            out_height = ceil(((height - kernel[0])/stride[0]) + 1)
            out_width = ceil(((width - kernel[1])/stride[1]) + 1)

        (Replace 'ceil' with 'floor' if ceil_mode is False.)
    """
    # Expand kernel/stride size shorthands
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    # If the kernel is larger than the input then there are enough tricks
    # needed below to make it not worth handling this edge-case until I
    # actually need to...
    if kernel[0] > x.shape[-2]:
        raise ValueError("Kernel taller than input.")
    if kernel[1] > x.shape[-1]:
        raise ValueError("Kernel wider than input.")

    # We don't have any sensible handling for when ceil_mode is used with a
    # kernel smaller than the step -- what happens in this case is somewhat
    # ill-defined anyway...
    if ceil_mode:
        if stride[0] > kernel[0]:
            raise ValueError("Stride taller than kernel.")
        if stride[1] > kernel[1]:
            raise ValueError("Stride wider than kernel.")

    # Get a windowed view of 'x's final two dimensions.
    #
    # NB: Regardless of the ceil_mode setting, this includes only 'complete'
    # kernel inputs where no padding is required. Padded cases are handled
    # specially later.
    #
    # windowed.shape == (..., wh, ww, kernel[0], kernel[1])
    windowed = np.lib.stride_tricks.sliding_window_view(
        x,
        window_shape=kernel,
        axis=(-2, -1),
    )  # type: ignore
    # XXX: Unclear why MyPy doesn't like the type of 'x' above here...

    # Pull out required strides (again ignoring 'incomplete' windows)
    #
    # (..., out_height_complete, out_width_complete, kernel[0], kernel[1])
    windowed = windowed[..., :: stride[0], :: stride[1], :, :]

    # Create the output array (including an extra row or column as needed
    # when ceil_mode is True).
    #
    # (..., out_height, out_width)
    rounding_mode = ceil if ceil_mode else floor
    out_height = rounding_mode(((x.shape[-2] - kernel[0]) / stride[0]) + 1)
    out_width = rounding_mode(((x.shape[-1] - kernel[1]) / stride[1]) + 1)
    if out is None:
        out = np.zeros(shape=(x.shape[:-2] + (out_height, out_width)), dtype=x.dtype)
    else:
        # Sanity check user provided correctly shaped output array
        assert out.shape == x.shape[:-2] + (out_height, out_width)

    # We now perform max pooling for all complete kernels
    np.amax(
        windowed,
        axis=(-2, -1),
        out=out[..., : windowed.shape[-4], : windowed.shape[-3]],
    )  # (..., out_height_complete, out_width_complete)

    # Now we perform max pooling for any overspilled kernels when required in
    # ceil_mode
    if ceil_mode:
        # When some partial kernels are required, find their size
        bottommost_kernel_end = ((out.shape[-2] - 1) * stride[0]) + kernel[0]
        rightmost_kernel_end = ((out.shape[-1] - 1) * stride[1]) + kernel[1]

        stragglers_bottom = stragglers_right = 0
        if bottommost_kernel_end > x.shape[-2]:
            stragglers_bottom = x.shape[-2] - (bottommost_kernel_end - kernel[0])
        if rightmost_kernel_end > x.shape[-1]:
            stragglers_right = x.shape[-1] - (rightmost_kernel_end - kernel[1])

        # Process straggler rows and columns which were not processed by any
        # full-sized kernel.
        if stragglers_bottom:
            max_pool_2d(
                x[..., -stragglers_bottom:, :],
                kernel=(stragglers_bottom, kernel[1]),
                stride=stride,
                ceil_mode=False,
                out=out[..., -1:, : windowed.shape[-3]],
            )

        if stragglers_right:
            max_pool_2d(
                x[..., :, -stragglers_right:],
                kernel=(kernel[0], stragglers_right),
                stride=stride,
                ceil_mode=False,
                out=out[..., : windowed.shape[-4], -1:],
            )

        if stragglers_bottom and stragglers_right:
            np.max(
                x[..., -stragglers_bottom:, -stragglers_right:],
                axis=(-2, -1),
                out=out[..., -1, -1],
            )
    else:
        # Sanity check when ceil_mode is False
        assert out.shape == windowed.shape[:-2]

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
