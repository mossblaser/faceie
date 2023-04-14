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


def conv2d(
    img: NDArray,
    weights: NDArray,
    biases: NDArray | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> NDArray:
    """
    Perform 2D convolution on the provided image using the provided kernel
    weights and biases.

    Parameters
    ==========
    img : array (in_channels, img_height, img_width) or (num_batches, in_channels, img_height, img_width)
        The input image, or a batch of images to process simultaneously.
    weights : array (out_channels, in_channels, kernel_height, kernel_width)
        The kernel to apply to the input.
    biases : array (out_channels, ) or None
        The biases to add to each output channel -- or None to not add biases.
    stride : int or (int, int)
        The stride of the convolutional filter.
    padding : int or (int, int)
        The (zero) padding to add to the top-and-bottom and left-and-right of
        the input. When non-zero, stride must be 1.

    Returns
    =======
    array (out_channels, out_height, out_width) or (num_batches, out_channels, out_height, out_width)
        The convolution output.

        The num_batches dimension will be present iff it it was included in the
        input img.
    """
    # Expand stride and padding to pairs
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    # Sanity check stride/padding not used simultaneously
    if (stride[0] != 1 or stride[1] != 1) and (padding[0] != 0 or padding[1] != 0):
        raise ValueError("Padding not supported when stride is not 1.")

    # Internally always treat the input as batched
    batched = len(img.shape) == 4
    if not batched:
        img = np.expand_dims(img, 0)

    num_batches = img.shape[0]
    out_channels, in_channels, kernel_height, kernel_width = weights.shape

    # Produce a sliding window over the input image to which the kernel will be
    # applied.
    #
    # NB: It is unclear why mypy fails to find a suitable override of
    # sliding_window_view matching our (pretty vanilla) usage here. As such we
    # just ignore it for now...
    windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore
        img,  # (num_batches, in_channels, img_height, img_width)
        (kernel_height, kernel_width),
        axis=(-2, -1),
    )  # (num_batches, in_channels, out_height, out_width, kernel_height, kernel_width)

    # Apply stride
    windows = windows[:, :, :: stride[0], :: stride[1], :, :]

    # Create output array
    out_height, out_width = windows.shape[2:4]
    padded_out_height = out_height + (padding[0] * 2)
    padded_out_width = out_width + (padding[1] * 2)
    out = np.zeros(
        (num_batches, out_channels, padded_out_height, padded_out_width),
        dtype=windows.dtype,
    )

    # Apply the convolution kernel across all un-padded regions
    np.einsum(
        "bihwkl,oikl->bohw",
        # (num_batches:b, in_channels:i, out_height:h, out_width:w, kernel_height:k, kernel_width:l)
        windows,
        # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
        weights,
        # (num_batches, out_channels, out_height, out_width)
        out=out[
            :,
            :,
            padding[0] : padding[0] + out_height,
            padding[1] : padding[1] + out_width,
        ],
        optimize=True,  # Free 10x speedup :D
    )

    # Apply convolutions which overhang into the padding regions
    #
    # Rather than expanding the input image (requiring a copy) we run
    # convolutions with cropped kernels along the edges of the input to compute
    # the equivalent values. This regrettably requires handling each edge and
    # corner specially below leading to a fair bit of verbosity... Sorry.

    # Overhang top/bottom edges only
    for y_pad_offset in range(1, padding[0] + 1):
        windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore
            img,
            (kernel_height - y_pad_offset, kernel_width),
            axis=(-2, -1),
        )

        # Top-edge convolutions
        np.einsum(
            "biwkl,oikl->bow",
            # (num_batches:b, in_channels:i, out_width:w, kernel_height:k, kernel_width:l)
            windows[:, :, 0, :],
            # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
            weights[:, :, y_pad_offset:, :],
            out=out[
                :, :, padding[0] - y_pad_offset, padding[1] : padding[1] + out_width
            ],
            optimize=True,
        )

        # Bottom-edge convolutions
        np.einsum(
            "biwkl,oikl->bow",
            # (num_batches:b, in_channels:i, out_width:w, kernel_height:k, kernel_width:l)
            windows[:, :, -1, :],
            # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
            weights[:, :, :-y_pad_offset, :],
            out=out[
                :,
                :,
                -(padding[0] - y_pad_offset + 1),
                padding[1] : padding[1] + out_width,
            ],
            optimize=True,
        )

    # Overhang left/right edges only
    for x_pad_offset in range(1, padding[1] + 1):
        windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore
            img,
            (kernel_height, kernel_width - x_pad_offset),
            axis=(-2, -1),
        )

        # Top-edge convolutions
        np.einsum(
            "bihkl,oikl->boh",
            # (num_batches:b, in_channels:i, out_height:h, kernel_height:k, kernel_width:l)
            windows[:, :, :, 0],
            # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
            weights[:, :, :, x_pad_offset:],
            out=out[
                :, :, padding[0] : padding[0] + out_height, padding[1] - x_pad_offset
            ],
            optimize=True,
        )

        # Bottom-edge convolutions
        np.einsum(
            "bihkl,oikl->boh",
            # (num_batches:b, in_channels:i, out_height:h, kernel_height:k, kernel_width:l)
            windows[:, :, :, -1],
            # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
            weights[:, :, :, :-x_pad_offset],
            out=out[
                :,
                :,
                padding[0] : padding[0] + out_height,
                -(padding[1] - x_pad_offset + 1),
            ],
            optimize=True,
        )

    # Overhang in corners
    for y_pad_offset in range(1, padding[0] + 1):
        for x_pad_offset in range(1, padding[1] + 1):
            windows = np.lib.stride_tricks.sliding_window_view(  # type: ignore
                img,
                (kernel_height - y_pad_offset, kernel_width - x_pad_offset),
                axis=(-2, -1),
            )

            # Top-left corner convolution
            np.einsum(
                "bikl,oikl->bo",
                # (num_batches:b, in_channels:i, kernel_height:k, kernel_width:l)
                windows[:, :, 0, 0],
                # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
                weights[:, :, y_pad_offset:, x_pad_offset:],
                out=out[:, :, padding[0] - y_pad_offset, padding[1] - x_pad_offset],
                optimize=True,
            )

            # Top-right corner convolution
            np.einsum(
                "bikl,oikl->bo",
                # (num_batches:b, in_channels:i, kernel_height:k, kernel_width:l)
                windows[:, :, 0, -1],
                # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
                weights[:, :, y_pad_offset:, :-x_pad_offset],
                out=out[
                    :, :, padding[0] - y_pad_offset, -(padding[1] - x_pad_offset + 1)
                ],
                optimize=True,
            )

            # Bottom-left corner convolution
            np.einsum(
                "bikl,oikl->bo",
                # (num_batches:b, in_channels:i, kernel_height:k, kernel_width:l)
                windows[:, :, -1, 0],
                # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
                weights[:, :, :-y_pad_offset, x_pad_offset:],
                out=out[
                    :, :, -(padding[0] - y_pad_offset + 1), padding[1] - x_pad_offset
                ],
                optimize=True,
            )

            # Bottom-right corner convolution
            np.einsum(
                "bikl,oikl->bo",
                # (num_batches:b, in_channels:i, kernel_height:k, kernel_width:l)
                windows[:, :, -1, -1],
                # (out_channels:o, in_channels:i, kernel_size:k, kernel_size:l)
                weights[:, :, :-y_pad_offset, :-x_pad_offset],
                out=out[
                    :,
                    :,
                    -(padding[0] - y_pad_offset + 1),
                    -(padding[1] - x_pad_offset + 1),
                ],
                optimize=True,
            )

    # Add biases
    if biases is not None:
        out += np.expand_dims(biases, (1, 2))  # bases.shape = (out_channels, 1, 1)

    if batched:
        return out
    else:
        return out[0]


def relu(x: NDArray) -> NDArray:
    """
    Implements a Rectified Linear Unit (ReLU).

    Values in ``x`` are mapped to themselves if greater than zero or clamped to
    zero otherwise.
    """
    return np.maximum(x, 0)


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
