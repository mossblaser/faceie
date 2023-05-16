"""
Numpy based implementations of the main Inception-ResNet-v1/FaceNet model components.
"""

from typing import NamedTuple, Iterable, Callable

from PIL import Image

import numpy as np
from numpy.typing import NDArray

from pathlib import Path

from functools import lru_cache

import weightie

from faceie import __version__

from faceie.image_to_array import image_to_array

from faceie.nn import (
    linear,
    conv2d,
    relu,
    batch_normalisation_2d,
    max_pool_2d,
    PaddingType,
    l2_normalisation,
)

from faceie.nn_arguments import (
    LinearWeights,
    Conv2DWeights,
    BatchNormalisationWeights,
)


class ConvolutionalUnitWeights(NamedTuple):
    """
    Convolutional kernel and batch normalisation for use by the
    :py:func:`convolutional_unit` unit.
    """

    # Convolutional kernel weights
    #
    # (out_channels, in_channels, kernel_height, kernel_width)
    kernel: NDArray

    # Batch normalisation
    batch_normalisation: BatchNormalisationWeights


class StemWeights(NamedTuple):
    """
    Convolution weights used by the stem network.
    """

    conv_1: ConvolutionalUnitWeights
    conv_2: ConvolutionalUnitWeights
    conv_3: ConvolutionalUnitWeights
    conv_4: ConvolutionalUnitWeights
    conv_5: ConvolutionalUnitWeights
    conv_6: ConvolutionalUnitWeights


class InceptionResNetAWeights(NamedTuple):
    """
    Weights used for an Inception-ResNet-A module.
    """

    branch_1_conv: ConvolutionalUnitWeights

    branch_2_conv_1: ConvolutionalUnitWeights
    branch_2_conv_2: ConvolutionalUnitWeights

    branch_3_conv_1: ConvolutionalUnitWeights
    branch_3_conv_2: ConvolutionalUnitWeights
    branch_3_conv_3: ConvolutionalUnitWeights

    output_conv: Conv2DWeights

    # NB: This scaling factor is appled to the output of the above convolutions
    # prior to adding the input values. That is, smaller numbers reduce the
    # influence of the convolution results relative to the input in the final
    # output.
    residual_scale: float


class ReductionAWeights(NamedTuple):
    """
    Weights used for the Reduction-A module.
    """

    branch_1_conv: ConvolutionalUnitWeights

    branch_2_conv_1: ConvolutionalUnitWeights
    branch_2_conv_2: ConvolutionalUnitWeights
    branch_2_conv_3: ConvolutionalUnitWeights


class InceptionResNetBWeights(NamedTuple):
    """
    Weights used for an Inception-ResNet-B module.
    """

    branch_1_conv: ConvolutionalUnitWeights

    branch_2_conv_1: ConvolutionalUnitWeights
    branch_2_conv_2: ConvolutionalUnitWeights
    branch_2_conv_3: ConvolutionalUnitWeights

    output_conv: Conv2DWeights

    # NB: This scaling factor is appled to the output of the above convolutions
    # prior to adding the input values. That is, smaller numbers reduce the
    # influence of the convolution results relative to the input in the final
    # output.
    residual_scale: float


class ReductionBWeights(NamedTuple):
    """
    Weights used for the Reduction-B module.
    """

    branch_1_conv_1: ConvolutionalUnitWeights
    branch_1_conv_2: ConvolutionalUnitWeights

    branch_2_conv_1: ConvolutionalUnitWeights
    branch_2_conv_2: ConvolutionalUnitWeights

    branch_3_conv_1: ConvolutionalUnitWeights
    branch_3_conv_2: ConvolutionalUnitWeights
    branch_3_conv_3: ConvolutionalUnitWeights


class InceptionResNetCWeights(NamedTuple):
    """
    Weights used for an Inception-ResNet-C module.
    """

    branch_1_conv: ConvolutionalUnitWeights

    branch_2_conv_1: ConvolutionalUnitWeights
    branch_2_conv_2: ConvolutionalUnitWeights
    branch_2_conv_3: ConvolutionalUnitWeights

    output_conv: Conv2DWeights

    # NB: This scaling factor is appled to the output of the above convolutions
    # prior to adding the input values. That is, smaller numbers reduce the
    # influence of the convolution results relative to the input in the final
    # output.
    residual_scale: float


class FaceNetWeights(NamedTuple):
    """Weights for the complete face-to-embedding function."""

    stem: StemWeights

    inception_resnet_a: list[InceptionResNetAWeights]
    reduction_a: ReductionAWeights
    inception_resnet_b: list[InceptionResNetBWeights]
    reduction_b: ReductionBWeights
    inception_resnet_c: list[InceptionResNetCWeights]

    output_dimension_reduction: LinearWeights
    output_batch_normalisation: BatchNormalisationWeights


# NB: This cache means that when load is called repeatedly and automatically by
# the encode function below it will not actually require us to re-load the
# weights file.
#
# NB: since the returned data is mmaped anyway we don't need to worry too much
# about keeping the weights loaded beyond their useful lifetime as the OS can
# swap them out if they're getting in the way anyway.
@lru_cache(maxsize=1)
def load(
    source: Path | str = "20180402-114759-vggface2.weights",
    search_paths: list[Path] | None = None,
    update: bool = False,
    progress_callback: Callable[[list[str], str, int, int | None], None]
    | None = weightie.downloader.print_status,
    min_callback_interval: float = 0.5,
) -> FaceNetWeights:
    """
    Load a set of weights from a file (if a Path is given) or automatically
    download a named weights file from the GitHub release (if a string is
    given).

    Parameters
    ==========
    source : local weights file (Path) or GitHub asset filename (str)
        The weights to be downloaded.
    search_paths : [Path, ...] or None
        When the source is a GitHub asset filename, a list of locations to
        search for weights locally before resorting to downloading the file. If
        None is given, will search platform-specific data directories.

        When downloading weights, the first item on the search path (by default
        the user application data directory) will be used to store the
        downloaded weights.
    update : bool
        If True, force a check for new weights to download.
    progress_callback : f(list_of_files, file, bytes_downloaded, bytes_total) or None
        During file downloads this will be called every min_callback_interval
        seconds with the status of the download. By default this will print
        status to stderr. Disable by passing None.
    min_callback_interval : float
        See progress_callback.

    Returns
    =======
    Weights
        The model weights.

        Loaded weights are memory mmapped meaning that the data will not
        actually be read from disk until it is used and may be freely swapped
        out of RAM by the OS when needed if they're not being used.
    """
    if isinstance(source, str):
        source = weightie.download(
            repository="mossblaser/faceie",
            asset_filenames=[source],
            target_version=__version__,
            search_paths=search_paths,
            update=update,
            progress_callback=progress_callback,
            min_callback_interval=min_callback_interval,
        )[source]

    return weightie.load(source.open("rb"))


def convolutional_unit(
    x: NDArray,
    weights: ConvolutionalUnitWeights,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | PaddingType = 0,
) -> NDArray:
    """
    Implements the base convolutional unit used throughout Inception-style
    networks::

                   +--------+      +------------+      +------+
        Input ---> | Conv2D | ---> | Batch Norm | ---> | ReLU | ---> Output
                   +--------+      +------------+      +------+

    Parameters
    ==========
    x : array (num_batches, in_channels, height, width)
    weights : ConvolutionalUnitWeights
    stride : int or (int, int)
        The stride of the convolution operation.
    padding : int or (int, int) or PaddingType
        The padding of the convolution operation.

    Returns
    =======
    array (batch, out_channels, height, width)
    """
    # NB: The convolution does not include a bias since the batch normalisation
    # includes its own bias.
    x = conv2d(x, weights.kernel, stride=stride, padding=padding)

    # TODO: The batch normalisation essentially boils down to a single
    # multiply-add. We could potentially fold this into the convolution weights
    # (and add some biases) by adjusting the weights. Something to consider!
    x = batch_normalisation_2d(x, *weights.batch_normalisation, channel_axis=1)

    x = relu(x)

    return x


def stem(img: NDArray, weights: StemWeights) -> NDArray:
    """
    Implements the 'stem' of the network as described by figure 14 of the
    Inception ResNet paper.

    The stem is intended to quickly drop the number of pixels (to about a 64th
    of the input) whist increasing the channel count (to 256).  This might be
    interpreted as performing low-level feature extraction from the input.

    Parameters
    ==========
    img : array (num_batches, 3, height, width)
        The input image(s).
    weights : StemWeights

    Returns
    =======
    array (num_batches, 256, out_height, out_width)
        A processed version of the input image with approximately a 64th of the
        number of pixels but 256 dimensions rather than three.

        The output dimensions are given by the following formulae:

        * out_height = ((img_height - 19) // 8)
        * out_width = ((img_width - 19) // 8)
    """
    x = img

    # 3x3 convolution, 3->32 channels, stride 2, no padding ('V')
    #
    # x_height = ((img_height - 3) // 2) + 1    # See conv2d output dim. formulae
    #             (img_height - 1) // 2
    # NB: x_width formulae are similar since all convolutions are square
    x = convolutional_unit(
        x, weights.conv_1, stride=2
    )  # (num_batches, 32, x_height, x_width)

    # 3x3 convolution, 32->32 channels, no padding ('V')
    #
    # x_height = ((img_height - 1) // 2) - 3 + 1
    #            ((img_height - 1) // 2) - 2
    #             (img_height - 5) // 2
    x = convolutional_unit(x, weights.conv_2)  # (num_batches, 32, x_height, x_width)

    # 3x3 convolution, 32->64 channels, zero padded to input size
    #
    # x_height = (img_height - 5) // 2    # Unchanged
    #
    # (num_batches, 64, x_height, x_width)
    x = convolutional_unit(x, weights.conv_3, padding=PaddingType.same)

    # 3x3 max pooling, stride 2, no padding ('V')
    #
    # x_height = ((((img_height - 5) // 2) - 3) // 2) + 1
    #             (((img_height - 5) // 2) - 1) // 2
    #             ( (img_height - 7) // 2)      // 2
    #               (img_height - 7) // 4
    x = max_pool_2d(x, kernel=3, stride=2)  # (num_batches, 64, x_height, x_width)

    # 1x1 convolution, 64->80 channels
    #
    # x_height = (img_height - 7) // 4
    #
    # (num_batches, 80, x_height, x_width)
    x = convolutional_unit(x, weights.conv_4, padding=PaddingType.same)

    # 3x3 convolution, 80->192 channels, no padding ('V')
    #
    # x_height =  (img_height - 7) // 4
    #            ((img_height - 7) // 4) - 3 + 1
    #            ((img_height - 7) // 4) - 2
    #             (img_height - 15) // 4
    x = convolutional_unit(x, weights.conv_5)  # (num_batches, 192, x_height, x_width)

    # 3x3 convolution, 192->256 channels, stride 2, no padding ('V')
    #
    # x_height =    (img_height - 15) // 4
    #            ((((img_height - 15) // 4) - 3) // 2) + 1
    #            ((((img_height - 15) // 4) - 1) // 2)
    #            ( ((img_height - 19) // 4)      // 2)
    #               (img_height - 19) // 8
    x = convolutional_unit(
        x, weights.conv_6, stride=2
    )  # (num_batches, 256, x_height, x_width)

    return x


def inception_resnet_a(img: NDArray, weights: InceptionResNetAWeights) -> NDArray:
    """
    Implements the Inception-ResNet-A module as described by figure 10 of the
    Inception ResNet paper.

    Parameters
    ==========
    img : array (num_batches, 256, height, width)
        The input image(s).
    weights : InceptionResNetAWeights

    Returns
    =======
    array (num_batches, 256, height, width)
        A processed version of the input with the same size and number of
        channels.
    """
    # Branch 1 (1x1 convolution)
    branch_1 = img
    # 1x1 convolution, 256->32 channels
    branch_1 = convolutional_unit(
        branch_1, weights.branch_1_conv, padding=PaddingType.same
    )

    # Branch 2 (3x3 convolution)
    branch_2 = img
    # 1x1 convolution, 256->32 channels
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_1, padding=PaddingType.same
    )
    # 3x3 convolution, 32->32 channels, zero padded to input size
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_2, padding=PaddingType.same
    )

    # Branch 3 (factorised 5x5 convolution)
    branch_3 = img
    # 1x1 convolution, 256->32 channels
    branch_3 = convolutional_unit(
        branch_3, weights.branch_3_conv_1, padding=PaddingType.same
    )
    # 3x3 convolution, 32->32 channels, zero padded to input size
    branch_3 = convolutional_unit(
        branch_3, weights.branch_3_conv_2, padding=PaddingType.same
    )
    # 3x3 convolution, 32->32 channels, zero padded to input size
    branch_3 = convolutional_unit(
        branch_3, weights.branch_3_conv_3, padding=PaddingType.same
    )

    # Combine branches
    output = np.concatenate(
        (branch_1, branch_2, branch_3), axis=1
    )  # (num_batches, 96, h, w)

    # 1x1 convolution, 96->256 channels
    output = conv2d(output, *weights.output_conv, padding=PaddingType.same)

    # Residual connection
    output *= weights.residual_scale
    output += img

    # Final ReLU
    output = relu(output)

    return output


def reduction_a(x: NDArray, weights: ReductionAWeights) -> NDArray:
    """
    Implements the Reduction-A module as described by figure 7 and table 1 of
    the Inception ResNet paper.

    This reduces the spatial resolution of the input whilst increasing the
    number of channels.

    Parameters
    ==========
    x : array (num_batches, 256, height, width)
        The input (as output by a :py:func:`inception_resnet_a` module).
    weights : ReductionAWeights

    Returns
    =======
    array (num_batches, 896, out_height, out_width)
        A processed version of the input with approximately a quarter of the
        number of pixels and 3.5 times the number of channels.

        The output dimensions are, more precisely, given by the formulae:

        * out_height = (height - 1) // 2
        * out_width = (width - 1) // 2
    """

    # NB: In this implementation, the branches are re-ordered with respect to
    # figure 7 to ensure weight compatibility with FaceNet-PyTorch. In the
    # paper, the max_poolling layer comes first, in FaceNet-PyTorch (and here)
    # it comes last.

    # Branch 1: 3x3 Convolution, 256 -> 384 channels, stride 2
    #
    # (num_batches, 384, out_height, out_width)
    branch_1 = convolutional_unit(x, weights.branch_1_conv, stride=2)

    # Branch 2: (factorised) 5x5 Convolution, 256 -> 256, stride 2
    #
    # 1x1 convolution, 256->192 channels
    # (num_batches, 192, height, width)
    branch_2 = convolutional_unit(x, weights.branch_2_conv_1, padding=PaddingType.same)
    # 3x3 convolution, 192->192 channels, zero padded to input size
    # (num_batches, 192, height, width)
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_2, padding=PaddingType.same
    )
    # 3x3 convolution, 192->256 channels, stride 2
    # (num_batches, 256, out_height, out_width)
    branch_2 = convolutional_unit(branch_2, weights.branch_2_conv_3, stride=2)

    # Branch 3: Simple max pooling with stride 2
    branch_3 = max_pool_2d(
        x, kernel=3, stride=2
    )  # (num_batches, 256, out_height, out_width)

    # (num_batches, 896, out_height, out_width)
    output = np.concatenate((branch_1, branch_2, branch_3), axis=1)

    return output


def inception_resnet_b(x: NDArray, weights: InceptionResNetBWeights) -> NDArray:
    """
    Implements the Inception-ResNet-B module as described by figure 11 of the
    Inception ResNet paper.

    Parameters
    ==========
    x : array (num_batches, 896, height, width)
        The input (as output by a :py:func:`reduction_a` module).
    weights : InceptionResNetBWeights

    Returns
    =======
    array (num_batches, 896, height, width)
        A processed version of the input with the same size and number of
        channels.
    """
    # Branch 1 (1x1 convolution)
    branch_1 = x
    # 1x1 convolution, 896->128 channels
    branch_1 = convolutional_unit(
        branch_1, weights.branch_1_conv, padding=PaddingType.same
    )

    # Branch 2 (factorised 7x7 convolution)
    branch_2 = x
    # 1x1 convolution, 896->128 channels
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_1, padding=PaddingType.same
    )
    # 1x7 convolution, 128->128 channels, zero padded to input size
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_2, padding=PaddingType.same
    )
    # 7x1 convolution, 128->128 channels, zero padded to input size
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_3, padding=PaddingType.same
    )

    # Combine branches
    output = np.concatenate((branch_1, branch_2), axis=1)  # (num_batches, 256, h, w)

    # 1x1 convolution, 256->896 channels
    output = conv2d(output, *weights.output_conv, padding=PaddingType.same)

    # Residual connection
    output *= weights.residual_scale
    output += x

    # Final ReLU
    output = relu(output)

    return output


def reduction_b(x: NDArray, weights: ReductionBWeights) -> NDArray:
    """
    Implements the Reduction-B module as described by figure 12 of the
    Inception ResNet paper.

    This further reduces the spatial resolution and increases the channel count
    of the input.

    Parameters
    ==========
    x : array (num_batches, 896, height, width)
        The input (as output by a :py:func:`inception_resnet_b` module).
    weights : ReductionBWeights

    Returns
    =======
    array (num_batches, 1792, out_height, out_width)
        A processed version of the input with approximately a quarter of the
        number of pixels and double the number of channels.

        The output dimensions are, more precisely, given by the formulae:

        * out_height = (height - 1) // 2
        * out_width = (width - 1) // 2
    """

    # NB: In this implementation, the branches are re-ordered with respect to
    # figure 12 to ensure weight compatibility with FaceNet-PyTorch. In the
    # paper, the max_poolling layer comes first, in FaceNet-PyTorch (and here)
    # it comes last.

    # Branch 1: 3x3 Convolution, 896 -> 384 channels, stride 2
    #
    # 1x1 convolution, 896->256 channels
    # (num_batches, 256, height, width)
    branch_1 = convolutional_unit(x, weights.branch_1_conv_1, padding=PaddingType.same)
    # 3x3 convolution, 256->384 channels, stride 2
    # (num_batches, 384, out_height, out_width)
    branch_1 = convolutional_unit(branch_1, weights.branch_1_conv_2, stride=2)

    # Branch 2: 3x3 Convolution, 896 -> 256 channels, stride 2
    #
    # 1x1 convolution, 896->256 channels
    # (num_batches, 256, height, width)
    branch_2 = convolutional_unit(x, weights.branch_2_conv_1, padding=PaddingType.same)
    # 3x3 convolution, 256->256 channels, stride 2
    # (num_batches, 256, out_height, out_width)
    branch_2 = convolutional_unit(branch_2, weights.branch_2_conv_2, stride=2)

    # Branch 3: (factorised) 5x5 Convolution, 256 -> 256, stride 2
    #
    # 1x1 convolution, 896->256 channels
    # (num_batches, 256, height, width)
    branch_3 = convolutional_unit(x, weights.branch_3_conv_1, padding=PaddingType.same)
    # 3x3 convolution, 256->256 channels, zero padded to input size
    # (num_batches, 256, height, width)
    branch_3 = convolutional_unit(
        branch_3, weights.branch_3_conv_2, padding=PaddingType.same
    )
    # 3x3 convolution, 256->256 channels, stride 2
    # (num_batches, 256, out_height, out_width)
    branch_3 = convolutional_unit(branch_3, weights.branch_3_conv_3, stride=2)

    # Branch 4: Simple max pooling with stride 2
    branch_4 = max_pool_2d(
        x, kernel=3, stride=2
    )  # (num_batches, 896, out_height, out_width)

    # (num_batches, 1792, out_height, out_width)
    output = np.concatenate((branch_1, branch_2, branch_3, branch_4), axis=1)

    return output


def inception_resnet_c(
    x: NDArray, weights: InceptionResNetCWeights, skip_relu: bool = False
) -> NDArray:
    """
    Implements the Inception-ResNet-C module as described by figure 13 of the
    Inception ResNet paper.

    Parameters
    ==========
    x : array (num_batches, 1792, height, width)
        The input (as output by a :py:func:`reduction_b` module).
    weights : InceptionResNetCWeights
    skip_relu : bool
        If True, the final ReLU operation prior to the output is omitted.

    Returns
    =======
    array (num_batches, 1792, height, width)
        A processed version of the input with the same size and number of
        channels.
    """
    # Branch 1 (1x1 convolution)
    branch_1 = x
    # 1x1 convolution, 1792->192 channels
    branch_1 = convolutional_unit(
        branch_1, weights.branch_1_conv, padding=PaddingType.same
    )

    # Branch 2 (factorised 3x3 convolution)
    branch_2 = x
    # 1x1 convolution, 1792->192 channels
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_1, padding=PaddingType.same
    )
    # 1x3 convolution, 192->192 channels, zero padded to input size
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_2, padding=PaddingType.same
    )
    # 3x1 convolution, 192->192 channels, zero padded to input size
    branch_2 = convolutional_unit(
        branch_2, weights.branch_2_conv_3, padding=PaddingType.same
    )

    # Combine branches
    output = np.concatenate((branch_1, branch_2), axis=1)  # (num_batches, 384, h, w)

    # 1x1 convolution, 384->1792 channels
    output = conv2d(output, *weights.output_conv, padding=PaddingType.same)

    # Residual connection
    output *= weights.residual_scale
    output += x

    # Final ReLU
    if not skip_relu:
        output = relu(output)

    return output


def encode_faces(
    image: Iterable[Image.Image] | Image.Image,
    weights: FaceNetWeights | None = None
) -> NDArray:
    """
    Produce the embeddings for a face, or series of faces.

    This implementation uses an adaption of the Inception-ResNet-v1
    architecture whose output stages are adapted to match the FaceNet
    architecture.

    Parameters
    ==========
    image : Image, [Image, ...]
        The face image or images.

        According to the Inception-ResNet-v1 paper, images should be 299x299
        pixels however FaceNet-PyTorch (which this implementation uses weights
        from) was trained on 160x160 images. As such, better results may be
        expected with the 160x160 images.
    weights : FaceNetWeights or None
        If None, a default set of weights will be downloaded using
        :py:func:`load`.

    Returns
    =======
    (512, ) or (num_batches, 512)
        A 512 dimensional embedding of the face or faces provided (depending on
        the shape of the input image.
    """
    if weights is None:
        weights = load()
    
    # Convert input to images
    if isinstance(image, Image.Image):
        x = image_to_array(image)
    else:
        x = np.stack(list(map(image_to_array, image)))

    # Force to (num_batches, 3, height, width)
    batched = x.ndim == 4
    if not batched:
        x = x.reshape(1, *x.shape)

    x = stem(x, weights.stem)

    for wa in weights.inception_resnet_a:
        x = inception_resnet_a(x, wa)

    x = reduction_a(x, weights.reduction_a)

    for wb in weights.inception_resnet_b:
        x = inception_resnet_b(x, wb)

    x = reduction_b(x, weights.reduction_b)

    # This implementation diverges from the Inception-ResNet-v1 paper at this
    # point in having an extra Inception-ResNet-C module without the ReLU stage
    # appended.
    for wc in weights.inception_resnet_c[:-1]:
        x = inception_resnet_c(x, wc)
    x = inception_resnet_c(x, weights.inception_resnet_c[-1], skip_relu=True)

    # "Adaptive average pooling" (or just taking the mean of all pixel values
    # in less fancy terms...)
    x = np.mean(x, axis=(2, 3))

    x = linear(x, *weights.output_dimension_reduction)

    x = batch_normalisation_2d(x, *weights.output_batch_normalisation, channel_axis=1)

    x = l2_normalisation(x, axis=1)

    if not batched:
        x = x[0]

    return x
