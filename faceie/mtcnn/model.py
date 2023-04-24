"""
This submodule implements the three deep convolutional networks from the MTCNN
face detection system. These are:

.. autofunction:: p_net
.. autofunction:: r_net
.. autofunction:: o_net

Also included in this module are the various datastructures used to hold the
model weights for these functions:

.. autoclass:: PNetWeights
.. autoclass:: RNetWeights
.. autoclass:: ONetWeights
.. autoclass:: LinearWeights
.. autoclass:: Conv2DWeights

The various ``*NetWeights`` classes include a ``load()`` class method which
loads a copy of these weights from the ``data`` subdirectory of this module.
The weights are simply stored pickled on disk since they are small enough not
to warrant any special handling.
"""

from typing import NamedTuple, Iterator, cast
from numpy.typing import NDArray

from functools import cache

import pickle

from pathlib import Path

import numpy as np

from faceie.nn import (
    linear,
    conv2d,
    max_pool_2d,
    softmax,
    prelu,
)

from faceie.nn_arguments import (
    LinearWeights,
    Conv2DWeights,
)


MODEL_DATA_DIR = Path(__file__).parent / "data"
"""
Directory containing precomputed model weights files.
"""


class PNetWeights(NamedTuple):
    """
    Weghts for the :py:func:`p_net` 'proposal' network.
    """

    # First 3x3 convolution 3-channels in, 10-channels out
    conv1: Conv2DWeights
    prelu1: NDArray

    # Second 3x3 convolution 10-channels in, 16-channels out
    conv2: Conv2DWeights
    prelu2: NDArray

    # Third 3x3 convolution 16-channels in, 32-channels out
    conv3: Conv2DWeights
    prelu3: NDArray

    # Output 1x1 'convolutions' (really just simple matrix multiplies)
    classifier: Conv2DWeights  # 32-channels in, 2 channels out
    bounding_boxes: Conv2DWeights  # 32-channels in, 4 channels out

    @classmethod
    @cache
    def load(cls) -> "PNetWeights":
        filename = MODEL_DATA_DIR / "p_net_weights.pickle"
        return cast(PNetWeights, pickle.load(filename.open("rb")))


class RNetWeights(NamedTuple):
    """
    Weghts for the :py:func:`r_net` 'proposal' network.
    """

    # First 3x3 convolution 3-channels in, 28-channels out
    conv1: Conv2DWeights
    prelu1: NDArray

    # Second 3x3 convolution 28-channels in, 48-channels out
    conv2: Conv2DWeights
    prelu2: NDArray

    # Third 3x3 convolution 48-channels in, 64-channels out
    conv3: Conv2DWeights
    prelu3: NDArray

    # Fully-connected linear stage 64x3x3 in, 128 out.
    linear: LinearWeights
    prelu4: NDArray

    # Final output weight matrices
    classifier: LinearWeights
    bounding_boxes: LinearWeights

    @classmethod
    @cache
    def load(cls) -> "RNetWeights":
        filename = MODEL_DATA_DIR / "r_net_weights.pickle"
        return cast(RNetWeights, pickle.load(filename.open("rb")))


class ONetWeights(NamedTuple):
    """
    Weghts for the :py:func:`o_net` 'proposal' network.
    """

    # First 3x3 convolution 3-channels in, 32-channels out
    conv1: Conv2DWeights
    prelu1: NDArray

    # Second 3x3 convolution 32-channels in, 64-channels out
    conv2: Conv2DWeights
    prelu2: NDArray

    # Third 3x3 convolution 64-channels in, 64-channels out
    conv3: Conv2DWeights
    prelu3: NDArray

    # Fourth 2x2 convolution 64-channels in, 128-channels out
    conv4: Conv2DWeights
    prelu4: NDArray

    # Fully-connected linear stage 128x3x3 in, 256 out.
    linear: LinearWeights
    prelu5: NDArray

    # Final output weight matrices
    classifier: LinearWeights
    bounding_boxes: LinearWeights
    landmarks: LinearWeights

    @classmethod
    @cache
    def load(cls) -> "ONetWeights":
        filename = MODEL_DATA_DIR / "o_net_weights.pickle"
        return cast(ONetWeights, pickle.load(filename.open("rb")))


def p_net(img, weights: PNetWeights | None = None) -> tuple[NDArray, NDArray]:
    """
    The 'proposal' network which uses a convolutional nerual network to very
    quickly (but crudely) locate all potential faces in the input image.

    The convolution uses a 12x12 kernel operates with a 2D stride of (2, 2) on
    an un-padded input meaning that the output resolution is a little under
    half that of the input.  Specifically, output dimensions are (input
    dimension - 10) // 2. (NB: Whilst the output is strieded, all input pixels
    are used during processing).

    To be explicit, the mapping between convolution result coordinates and
    input image regions is like so:

    * Output (0, 0) uses inputs (0, 0) to (11, 11) inclusive,
    * Output (0, 1) uses inputs (0, 2) to (11, 13) inclusive,
    * Output (1, 0) uses inputs (2, 0) to (13, 11) inclusive.
    * ...

    Parameters
    ==========
    img : array (3, height, width)
        The input image for processing. Pixel values should be given in the
        range -1.0 to 1.0.
    weights : PNetWeights or None
        If omitted, a default set of weights will be loaded automatically.

    Returns
    =======
    probabilities : array (out_height, out_width)
        A probability between 0.0 and 1.0 of there being a face in the
        convolved region.
    bounding_boxes : array (out_height, out_width, 4)
        The bounding boxes of the faces (if any) for each value in
        probabilities.

        The four values in dimension 2 are x1, y1, x2, y2 respectively.  These
        values are also scaled down such that '0' is the top-left of the
        corresponding input area and '1' is the bottom-right.

        See also :py:func:`resolve_p_net_bounding_boxes`.
    """
    if weights is None:
        weights = PNetWeights.load()

    # First (3x3) convolution stage
    x = conv2d(img, *weights.conv1)  # (10, height-2, width-2)

    # NB: The Zhang et al. paper does not actually specify any particular
    # non-linearity and defer details to "Multi-view face detection using deep
    # convolutional neural networks", Farfade et al. (2015). This paper in turn
    # cites "Imagenet classification with deep convolutional neural networks.",
    # Krizhevsky et al. (2012) -- better known as the AlexNet paper -- who
    # use ReLU non-linearities. Nevertheless, in their published Matlab code,
    # PReLU is used and so we do too!
    x = prelu(x, weights.prelu1, axis=0)

    # NB: The Zhang et al. paper specifies 3x3 max pooling here but the PyTorch
    # implementation uses a 2x2 kernel instead. We do the same here to keep the
    # implementations compatible.
    x = max_pool_2d(
        x, kernel=2, stride=2, ceil_mode=True
    )  # (10, (height-2) // 2, (width-2) // 2)

    # Second (3x3) convolution stage
    x = conv2d(x, *weights.conv2)  # (16, ((height-2) // 2) - 2, ((width-2) // 2) - 2)
    # (16, ((height-6) // 2),     ((width-6) // 2))
    x = prelu(x, weights.prelu2, axis=0)

    # Final (3x3) convolution stage
    x = conv2d(x, *weights.conv3)  # (32, ((height-6)  // 2) - 2, ((width-6)  // 2) - 2)
    # (32, ((height-10) // 2),     ((width-10) // 2))
    # (32, out_height,             out_width)
    x = prelu(x, weights.prelu3, axis=0)

    # After the prior convolution and maximum pooling steps, the resulting
    # convolution filter has a size of 12x12 and a step size of 2x2 (due to the
    # max pooling) as illustrated in figure 2 in the Zhang et al. paper.

    # Classification (i.e. face probability)
    #
    # NB: This is actually just a simple matrix multiply from each 32-element
    # vector to a 2-element vector (two logits representing not-face and
    # is-face cases, respectively). This is implemented here via conv2d for
    # convenience using a 1x1 kernel rather than faffing around shuffling
    # indices about.
    classification = conv2d(x, *weights.classifier)  # (2, out_height, out_width)

    # Convert from pair of logits to simple "is face" probability
    probabilities = softmax(classification, axis=0)[1]  # (out_height, out_width)

    # Bounding box calculation
    bounding_boxes = conv2d(x, *weights.bounding_boxes)  # (4, out_height, out_width)
    bounding_boxes = np.moveaxis(bounding_boxes, 0, 2)  # (out_height, out_width, 4)

    return (probabilities, bounding_boxes)


def r_net(img, weights: RNetWeights | None = None) -> tuple[NDArray, NDArray]:
    """
    The 'refinement' network which uses a convolutional nerual network to
    refine the bounding box around a face detected by the :py:func:`p_net`.

    Parameters
    ==========
    img : array (num_batches, 3, 24, 24)
        A series of 24x24 pixel input images for processing. Pixel values
        should be given in the range -1.0 to 1.0.
    weights : RNetWeights or None
        If omitted, a default set of weights will be loaded automatically.

    Returns
    =======
    probabilities : array (num_batches)
        A probability between 0.0 and 1.0 of there being a face in each input
        image.
    bounding_boxes : array (num_batches, 4)
        A refined bounding box for the face within each image.

        The four values in dimension 1 are x1, y1, x2, y2 respectively.
        These values are also scaled down such that '0' is the top-left of the
        input and '1' is the bottom-right.
    """
    if weights is None:
        weights = RNetWeights.load()

    # Sanity check input dimension
    assert img.shape[-3:] == (3, 24, 24)

    # First (3x3) convolution stage
    x = conv2d(img, *weights.conv1)  # (num_batches, 28, 22, 22)
    x = prelu(x, weights.prelu1, axis=1)  # See PReLU/ReLU note in p_net
    x = max_pool_2d(x, kernel=3, stride=2, ceil_mode=True)  # (num_batches, 28, 11, 11)

    # Second (3x3) convolution stage
    x = conv2d(x, *weights.conv2)  # (num_batches, 48, 9, 9)
    x = prelu(x, weights.prelu2, axis=1)
    x = max_pool_2d(x, kernel=3, stride=2, ceil_mode=True)  # (num_batches, 48, 4, 4)

    # Third (2x2) convolution stage
    #
    # NB: A typographical error in Fig 2. of the Zhang et al. paper appears to
    # indicate an output channel count of 64128, however a space is simply
    # mising between the '64' and '128'.
    x = conv2d(x, *weights.conv3)  # (num_batches, 64, 3, 3)
    x = prelu(x, weights.prelu3, axis=1)

    # Final 'fully connected' stage reduces the input images to 128-dimensional
    # vectors
    #
    # NB: In the PyTorch implementation (that we're aiming for weight
    # compatibility with), their matrix assumes our pixel data is ordered as
    # (width, height, channels) and they shuffle this data around accordingly
    # during processing.  Instead, we use a reorganised weight matrix so that
    # we can put in the (channels, height, width) ordered data we have
    # directly.
    x = x.reshape(x.shape[0], 64 * 3 * 3)  # (num_batches, 64*3*3)
    x = linear(x, *weights.linear)  # (num_batches, 128)
    x = prelu(x, weights.prelu4, axis=1)

    # Classification (i.e. face probability)
    classification = linear(x, *weights.classifier)  # (num_batches, 2)
    probabilities = softmax(classification, axis=-1)[..., 1]  # (num_batches, )

    # Bounding boxes
    bounding_boxes = linear(x, *weights.bounding_boxes)  # (num_batches, 4)

    return (probabilities, bounding_boxes)


def o_net(img, weights: ONetWeights | None = None) -> tuple[NDArray, NDArray, NDArray]:
    """
    The 'output' network which uses a convolutional nerual network to finally
    define the bounding box, facial features and probabilities for a face
    detected by the :py:func:`r_net`.

    Parameters
    ==========
    img : array (num_batches, 3, 48, 48)
        A series of 48x48 pixel input images for processing. Pixel values
        should be given in the range -1.0 to 1.0.
    weights : ONetWeights or None
        If omitted, a default set of weights will be loaded automatically.

    Returns
    =======
    probabilities : array (num_batches)
        A probability between 0.0 and 1.0 of there being a face in each input
        image.
    bounding_boxes : array (num_batches, 4)
        A refined bounding box for the face within each image.

        The four values in dimension 1 are x1, y1, x2, y2 respectively.
        These values are also scaled down such that '0' is the top-left of the
        input and '1' is the bottom-right.
    landmarks : array (num_batches, 10)
        The facial feature landmarks for the face within the image.

        The ten values in dimension 1 are x1, y1, x2, y2 and so on where:

        * (x1, y1) is the left eye coordinate
        * (x2, y2) is the right eye coordinate
        * (x3, y3) is the nose coordinate
        * (x4, y4) is the left mouth coordinate
        * (x5, y5) is the right mouth coordinate

        All values are scaled such that '1' is the width or height of the input
        bounding box and are relative to the top-left corner of the input.
    """
    if weights is None:
        weights = ONetWeights.load()

    # Sanity check input dimension
    assert img.shape[-3:] == (3, 48, 48)

    # First (3x3) convolution stage
    x = conv2d(img, *weights.conv1)  # (num_batches, 32, 46, 46)
    x = prelu(x, weights.prelu1, axis=1)  # See PReLU/ReLU note in p_net
    x = max_pool_2d(x, kernel=3, stride=2, ceil_mode=True)  # (num_batches, 32, 23, 23)

    # Second (3x3) convolution stage
    x = conv2d(x, *weights.conv2)  # (num_batches, 64, 21, 21)
    x = prelu(x, weights.prelu2, axis=1)
    x = max_pool_2d(x, kernel=3, stride=2, ceil_mode=True)  # (num_batches, 64, 10, 10)

    # Third (3x3) convolution stage
    x = conv2d(x, *weights.conv3)  # (num_batches, 64, 8, 8)
    x = prelu(x, weights.prelu3, axis=1)
    x = max_pool_2d(x, kernel=2, stride=2, ceil_mode=True)  # (num_batches, 64, 4, 4)

    # Fourth (2x2) convolution stage
    x = conv2d(x, *weights.conv4)  # (num_batches, 128, 3, 3)
    x = prelu(x, weights.prelu4, axis=1)

    # Final 'fully connected' stage reduces the input images to 256-dimensional
    # vectors.
    #
    # NB: Matrix re-ordered to match image memory layout compared with PyTorch
    # implementation -- see comment in r_net's fully connected stage.
    x = x.reshape(x.shape[0], 128 * 3 * 3)  # (num_batches, 128*3*3)
    x = linear(x, *weights.linear)  # (num_batches, 256)
    x = prelu(x, weights.prelu5, axis=1)

    # Classification (i.e. face probability)
    classification = linear(x, *weights.classifier)  # (num_batches, 2)
    probabilities = softmax(classification, axis=-1)[..., 1]  # (num_batches, )

    # Bounding boxes
    bounding_boxes = linear(x, *weights.bounding_boxes)  # (num_batches, 4)

    # Facial feature landmarks
    #
    # NB: Matrix is re-ordered compared with PyTorch implementation to
    # interleave x and y coordinates of the landmarks rather than having them
    # as 5 x coordinates followed by 5 y coordinates.
    landmarks = linear(x, *weights.landmarks)  # (num_batches, 10)

    return (probabilities, bounding_boxes, landmarks)
