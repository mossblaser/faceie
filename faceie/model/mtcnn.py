"""
An implementation of face recognition based on Multi-Task Cascaded
Convolutional Networks (MTCNN).

This technique was originally published by Zhang et al. in "Joint Face
Detection and Alignment using Multi-task Cascaded Convolutional Networks"
(2016).

This implementation attempts to be weight-compatible with Face-net PyTorch
(https://github.com/timesler/facenet-pytorch) but is otherwise an indepdendent
reimplementation. In particular, this means that PReLU non-linearities are used
(rather than ReLU).
"""


from typing import NamedTuple, Iterator, cast
from numpy.typing import NDArray

from PIL import Image

import math

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


MODEL_DATA_DIR = Path(__file__).parent / "data"
"""
Directory containing precomputed model weights files.
"""


class LinearWeights(NamedTuple):
    weights: NDArray
    biases: NDArray


class Conv2DWeights(NamedTuple):
    weights: NDArray
    biases: NDArray


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
    # use ReLU non-linearities.
    #
    # ...however, for weight compatibility with Face-net PyTorch, we use PReLU
    # instead.
    x = prelu(x, weights.prelu1, axis=0)
    
    # NB: The Zhang et al. paper specifies 3x3 max pooling here but the PyTorch
    # implementation uses a 2x2 kernel instead. We do the same here to keep the
    # implementations compatible.
    x = max_pool_2d(x, kernel=2, stride=2)  # (10, (height-2) // 2, (width-2) // 2)
    
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


def resolve_p_net_bounding_boxes(bounding_boxes: NDArray) -> NDArray:
    """
    Given the bounding box output of :py:func:`p_net`, resolve all bounding box
    coordinates to absolute pixel coordinates in the input image.
    
    Modifies the provided array in-place, but also returns it for convenience.
    """
    # The convolution kernel is 12x12 and so coordinates run from 0 to 11
    # inclusive -- here we scale-up from the nominal range of 0-1.
    bounding_boxes *= 11.0
    
    # Since the p_net convolution has a stride of 2x2, the output grid is half
    # the size so we multiply by two here to compensate.
    ys, xs = np.indices(bounding_boxes.shape[:2]) * 2
    bounding_boxes[..., 0::2] += np.expand_dims(xs, -1)
    bounding_boxes[..., 1::2] += np.expand_dims(ys, -1)
    
    return bounding_boxes


def non_maximum_suppression(
    probabilities: NDArray,
    bounding_boxes: NDArray,
    maximum_iou: float,
) -> NDArray:
    """
    Performs non-maximum suppression (NMS) on a collection of bounding boxes
    and associated probabilities. This returns the subset of bounding boxes
    which are both sufficiently non-overlapping.
    
    Parameters
    ==========
    probabilities : array (num_candidates)
        An array of probabilities of each candidate bounding box containing a
        face.
    bounding_boxes : array (num_candidates, 4)
        The corresponding bounding boxes which may contain a face, for the same
        shape of values as ``probabilities``. The last axis contains the
        bounding box corners as x1, y1, x2 and y2 respectively.
    maximum_iou : float
        The maximum overlapping area (as Intersection over Union (IoU)) two
        bounding boxes can have before the lower-probability box is removed.
    
    Returns
    =======
    array (num_results)
        An array of indices into the input ``probabilities`` and
        ``bounding_boxes`` arrays which indicate the bounding boxes selected by
        the non-maximum suppression algoirthm.
    """
    # We proceed by repeatedly selecting the most likely bounding box and then
    # removing any others which overlap too much with it.
    selected_indices = []
    candidate_indices = np.argsort(probabilities)
    while len(candidate_indices) > 0:
        # Find (and keep) most likley face
        best_index = candidate_indices[-1]
        selected_indices.append(best_index)
        candidate_indices = candidate_indices[:-1]
        
        # Now lets find (and remove) any faces which overlap with this face by
        # too much.
        
        # Bounding box of most likely ('Best') face
        bx1, by1, bx2, by2 = bounding_boxes[best_index]
        
        # Bounding boxes of all 'Other' candidate faces
        ox1s, oy1s, ox2s, oy2s = bounding_boxes[candidate_indices].T
        
        # Work out intersecting areas
        #
        # In 1D, consider two pairs of coordinates with an intersecting region:
        #
        #     ax1|----------|ax2
        #         bx1|--------------|bx2
        #         ix1|------|ix2
        #
        # The intersecting region (denoted by ix1 and ix2 above) is given by
        #
        #     ix1 = max(ax1, bx1)
        #     ix2 = min(ax2, bx2)
        #
        # There is an intersection whenever ix1 < ix2, otherwise the regions do
        # not intersect.
        #
        # So, lets carry this out en-masse on our bounding box vs all other
        # bounding boxes -- and for both x and y axes:
        intersection_x1s = np.maximum(bx1, ox1s)
        intersection_x2s = np.minimum(bx2, ox2s)
        intersection_xs = np.maximum(0, intersection_x2s - intersection_x1s)
        
        intersection_y1s = np.maximum(by1, oy1s)
        intersection_y2s = np.minimum(by2, oy2s)
        intersection_ys = np.maximum(0, intersection_y2s - intersection_y1s)
        
        intersection_areas = intersection_xs * intersection_ys
        
        # Now lets compute the total area of the union of the best bounding box
        # and all other bounding boxes.
        best_area = (bx2 - bx1) * (by2 - by1)
        other_areas = (ox2s - ox1s) * (oy2s - oy1s)
        union_areas = best_area + other_areas - intersection_areas
        
        # Finally, lets compute the intersection-over-union (IoU) and discard
        # any candidates with excessive overlap
        ious = intersection_areas / union_areas
        candidate_indices = candidate_indices[np.where(ious <= maximum_iou)]
    
    return np.array(selected_indices, dtype=int)

def make_square(bounding_boxes: NDArray, axis: int) -> NDArray:
    """
    Given an array of bounding boxes, enlarge these as necessary, centered on
    the existing centre point, to make them square.
    
    Parameters
    ==========
    bounding_boxes: array (..., 4, ...)
        An array which contains a 4-long dimension enumerating the x1, y1, x2
        and y2 coordinates of a bounding box.
    axis : int
        The index of the axis enumerating the bounding box coordinates.
    
    Returns
    =======
    array (same shape as bounding_boxes)
        A new bounding box array where all bounding boxes have x2-x1 == y2-y1.
    """
    # Make axis index positive
    axis %= bounding_boxes.ndim
    
    # Sanity check
    assert bounding_boxes.shape[axis] == 4
    
    # This function returns an index like (..., n, ...) with the 'n' on the
    # specified axis.
    I = lambda n: tuple(n if i == axis else slice(None) for i in range(bounding_boxes.ndim))
    IX1 = I(0)
    IY1 = I(1)
    IX2 = I(2)
    IY2 = I(3)
    
    x1s = bounding_boxes[IX1]
    y1s = bounding_boxes[IY1]
    x2s = bounding_boxes[IX2]
    y2s = bounding_boxes[IY2]
    
    # Find longest of width and height
    ws = x2s - x1s
    hs = y2s - y1s
    ls = np.maximum(ws, hs)
    
    # Compute centre of bounding boxes
    cxs = x1s + (ws * 0.5)
    cys = y1s + (hs * 0.5)
    
    half_ls = ls * 0.5
    
    # Compute new bounding boxes
    out = np.empty_like(bounding_boxes)
    out[IX1] = cxs - half_ls
    out[IY1] = cys - half_ls
    out[IX2] = cxs + half_ls
    out[IY2] = cys + half_ls
    
    return out


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
    x = max_pool_2d(x, kernel=3, stride=2)  # (num_batches, 28, 11, 11)
    
    # Second (3x3) convolution stage
    x = conv2d(x, *weights.conv2)  # (num_batches, 48, 9, 9)
    x = prelu(x, weights.prelu2, axis=1)
    x = max_pool_2d(x, kernel=3, stride=2)  # (num_batches, 48, 4, 4)
    
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


def resolve_coordinates(input_bounding_boxes: NDArray, coordinate_pairs: NDArray) -> NDArray:
    """
    Resolve coordinates (e.g. bounding box or landmark coordinates produced by
    the :py:func:`r_net` or :py:func:`o_net` functions) from 0-1 ranges to
    actual pixel coordinates.
    
    Parameters
    ==========
    input_bounding_boxes : array (num_batches, 4)
        For each entry in coordinate_pairs, the corresponding input pixel
        bounding box.
    coordinate_pairs : array (num_batches, even-number)
        An array of interleaved values like x1, y1, x2, y2, ... which will be
        scaled from nominal 0-1 ranges to actual pixel values.
        
        This array will be modified in-place.
    
    Returns
    =======
    Returns the ``coordinate_pairs`` array again.
    """
    widths = input_bounding_boxes[:, 2] - input_bounding_boxes[:, 0]
    heights = input_bounding_boxes[:, 3] - input_bounding_boxes[:, 1]
    
    # Scale offsets to input size
    coordinate_pairs[:, 0::2] *= np.expand_dims(widths, -1)
    coordinate_pairs[:, 1::2] *= np.expand_dims(heights, -1)
    
    # Translate into position
    coordinate_pairs[:, 0::2] += np.expand_dims(input_bounding_boxes[:, 0], -1)
    coordinate_pairs[:, 1::2] += np.expand_dims(input_bounding_boxes[:, 1], -1)
    
    return coordinate_pairs


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
    x = max_pool_2d(x, kernel=3, stride=2)  # (num_batches, 32, 23, 23)
    
    # Second (3x3) convolution stage
    x = conv2d(x, *weights.conv2)  # (num_batches, 64, 21, 21)
    x = prelu(x, weights.prelu2, axis=1)
    x = max_pool_2d(x, kernel=3, stride=2)  # (num_batches, 64, 10, 10)
    
    # Third (3x3) convolution stage
    x = conv2d(x, *weights.conv3)  # (num_batches, 64, 8, 8)
    x = prelu(x, weights.prelu3, axis=1)
    x = max_pool_2d(x, kernel=2, stride=2)  # (num_batches, 64, 4, 4)
    
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


class ImagePyramid:
    """
    Implements a simple image 'pyramid' containing multiple copies of an input
    image scaled down by a factor of two until a particular minimum size.
    """
    
    # The full set of images in the pyramid. The i-th image is a factor of 2^i
    # smaller than the input image.
    _images: list[Image.Image]
    
    def __init__(self, image: Image.Image, min_size: int, downscale_factor: float = 2.0) -> None:
        """
        Parameters
        ==========
        image : PIL Image
            The full-sized original image to use as the 'base' of the pyramid.
        downscale_factor : double
            The down-scaling factor applied to each successive image in the
            pyramdid.
        min_size : int
            The minimum height or width required for the smallest image in the
            pyramid. If the input is smaller than this a :py:exc:`ValueError`
            will be thrown.
        """
        if image.size[0] < min_size or image.size[1] < min_size:
            raise ValueError("Image smaller than minimum size.")
        
        assert downscale_factor > 1.0
        self.downscale_factor = downscale_factor
        
        # Create pyramid
        self._images = [image]
        while True:
            w, h = image.size
            w = round(w / self.downscale_factor)
            h = round(h / self.downscale_factor)
            if w < min_size or h < min_size:
                break
            
            image = image.resize((w, h))
            self._images.append(image)
    
    def __len__(self) -> int:
        """Get the number of levels in the pyramid."""
        return len(self._images)
    
    def __getitem__(self, idx: int) -> Image.Image:
        """Return the image at the given pyramid level."""
        return self._images[idx]
    
    def __iter__(self) -> Iterator[Image.Image]:
        return iter(self._images)
    
    def scale_between(self, level_a: int, level_b: int) -> float:
        """
        Return the scaling factor from pixel coordinates at level_a to
        equivalent pixel coordinates at level_b.
        """
        return self.downscale_factor ** (level_a - level_b)
    
    def closest_level(self, downscale_factor: float) -> int:
        """
        Return the level index whose downscale factor from the original image
        is as small as possible whilst being at least downscale_factor.
        """
        return min(
            len(self) - 1,
            max(0,
                math.floor(
                    math.log(downscale_factor) / math.log(self.downscale_factor)
                )
            )
        )
    
    def extract(
        self,
        crop: tuple[int, int, int, int], 
        size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """
        Extract a region within the image at the specified resolution.
        
        Parameters
        ==========
        crop : (x1, y1, x2, y2)
            The input region to extract (in full-resolution pixel coordinates).
            The x2 and y2 coordinates are exclusive: that is, these will be the
            first column and row *not* included in the crop.
        size : (width, height) or None
            If given, specifies the size to scale the output to. Otherwise,
            the native size is assumed.
        """
        x1, y1, x2, y2 = crop
        iw = x2 - x1
        ih = y2 - y1
        
        if size is None:
            size = (iw, ih)
        ow, oh = size
        
        # Pick the smallest scale from the pyramid at least as high resolution
        # as the output size to make the rescaling operation as small (and as
        # cheap) as possible.
        source_level = self.closest_level(min(iw / ow, ih / oh))
        
        level_scale_factor = self.scale_between(0, source_level)
        x1 *= level_scale_factor
        y1 *= level_scale_factor
        x2 *= level_scale_factor
        y2 *= level_scale_factor
        
        extract = self[source_level].crop((round(x1), round(y1), round(x2), round(y2)))
        return extract.resize((ow, oh))


def image_to_array(image: Image.Image) -> NDArray:
    """
    Convert an 8-bit RGB PIL image into a float32 Numpy array with shape (3,
    height, width) and valuesin the range -1 to +1.
    """
    out = np.asarray(image).astype(np.float32)
    
    # Rescale
    out -= 127.5
    out /= 128.0
    
    # Move channels to front
    out = np.moveaxis(out, 2, 0)
    
    return out


def get_proposals(
    image: NDArray,
    probability_threshold: float = 0.7,
) -> tuple[NDArray, NDArray]:
    """
    Given an image, run the :py:func:`p_net` proposal stage against an image,
    returning the probabilities and bounding boxes of all potential faces
    detected.
    
    Parameters
    ==========
    image : array (3, height, width)
        The image to process as values between -1 and +1. Must be at least
        12x12 pixels.
    probability_threshold : float
        The minimum probability of a face to accept as a proposal.
    
    Return
    ======
    array (num_proposals)
        The probabilities assigned to each found face.
    array (num_proposals, 4)
        The bounding box for each found face. Dimension 1 gives (x1, y1, x2,
        y2) in image pixel coordinates. The x2 and y2 coordinates are
        inclusive.
    """
    probs, bboxes = p_net(image)
    
    # Flatten
    shape = probs.shape
    probs = probs.reshape(np.product(shape))
    bboxes = bboxes.reshape(np.product(shape), 4)
    
    # Select only candidates with sufficiently high probability
    above_threshold = probs > probability_threshold
    probs = probs[above_threshold]
    bboxes = bboxes[above_threshold]
    
    # Resolve bounding boxes to pixel coordinates
    bboxes *= 11
    ys, xs = np.unravel_index(np.flatnonzero(above_threshold), shape)
    bboxes[:, 0::2] += np.expand_dims(xs, 1) * 2
    bboxes[:, 1::2] += np.expand_dims(ys, 1) * 2
    
    return (probs, bboxes)


def refine_proposals(
    pyramid: ImagePyramid,
    bounding_boxes: NDArray,
    probability_threshold: float = 0.7,
) -> tuple[NDArray, NDArray]:
    """
    Given a series of potential faces, return a more accurate probability and
    bounding box.
    
    Parameters
    ==========
    pyramid : ImagePyramid
        The image in which the faces reside.
    bounding_boxes : array (num_proposals, 4)
        The bounding boxes defining the locations of faces in the input image.
    probability_threshold : float
        The minimum (refined) probability of a face to return.
    
    Return
    ======
    probabilities : array (num_accepted_proposals)
        The (refined) probabilities of the input faces which surpass the given
        probability_threshold.
    bounding_boxes : array (num_accepted_proposals, 4)
        For each accepted face, a new refined bounding box.  Dimension 1 gives
        (x1, y1, x2, y2) in image pixel coordinates. The x2 and y2 coordinates
        are inclusive.
    """
    crops = make_square(bounding_boxes, axis=1)

    # Crop candidate faces from inputs
    faces = np.empty((crops.shape[0], 3, 24, 24), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(crops):
        faces[i] = image_to_array(pyramid.extract((x1, y1, x2 + 1, y2 + 1), (24, 24)))

    probs, bboxes = r_net(faces)

    # Convert to pixel coordinates
    bboxes = resolve_coordinates(crops, bboxes)

    # Select high-probability candidates
    selection = probs >= probability_threshold
    probs = probs[selection]
    bboxes = bboxes[selection]

    return probs, bboxes


def output_proposals(
    pyramid: ImagePyramid,
    bounding_boxes: NDArray,
    probability_threshold: float = 0.7,
) -> tuple[NDArray, NDArray]:
    """
    Given a series of potential faces, return a final (output) probability,
    bounding box and (optional) set of facial landmarks.
    
    Parameters
    ==========
    pyramid : ImagePyramid
        The image in which the faces reside.
    bounding_boxes : array (num_proposals, 4)
        The bounding boxes defining the locations of faces in the input image.
    probability_threshold : float
        The minimum (refined) probability of a face to return.
    
    Return
    ======
    probabilities : array (num_accepted_proposals)
        The (output) probabilities of the input faces which surpass the given
        probability_threshold.
    bounding_boxes : array (num_accepted_proposals, 4)
        For each accepted face, a new refined bounding box.  Dimension 1 gives
        (x1, y1, x2, y2) in image pixel coordinates. The x2 and y2 coordinates
        are inclusive.
    landmarks : array (num_accepted_proposals, 10)
        For each accepted face, the facial landmark coordinates. Dimension 1
        gives (x1, y1, x2, y2, ...) giving the coordinates of the left eye,
        right eye, nose, left mouth and right mouth.
    """
    crops = make_square(bounding_boxes, axis=1)

    # Crop candidate faces from inputs
    faces = np.empty((crops.shape[0], 3, 48, 48), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(crops):
        faces[i] = image_to_array(pyramid.extract((x1, y1, x2 + 1, y2 + 1), (48, 48)))

    probs, bboxes, landmarks = o_net(faces)

    # Convert to pixel coordinates
    bboxes = resolve_coordinates(crops, bboxes)
    landmarks = resolve_coordinates(crops, landmarks)

    # Select high-probability candidates
    selection = probs >= probability_threshold
    probs = probs[selection]
    bboxes = bboxes[selection]
    landmarks = landmarks[selection]

    return probs, bboxes, landmarks


class DetectedFaces(NamedTuple):
    probabilities: NDArray
    bounding_boxes: NDArray
    landmarks: NDArray


def detect_faces(
    image: Image.Image,
    pyramid_downscale_factor: float = np.sqrt(2),
    proposal_skip_downscale_factor: float = 4.0,
    proposal_probability_threshold: float = 0.7,
    refine_probability_threshold: float = 0.7,
    output_probability_threshold: float = 0.7,
    proposal_maximum_iou: float = 0.5,
    refine_maximum_iou: float = 0.5,
    output_maximum_iou: float = 0.5,
) -> DetectedFaces:
    """
    Detect faces within an image using the MTCNN algorithm by Zhang et al.
    
    Parameters
    ==========
    image : Image
        The image to detect faces within.
    pyramid_downscale_factor : float
        The image will be processed at a range of progressively sizes, related
        by this factor.
    proposal_skip_downscale_factor : float
        Skips running the proposal filtering process on images below this
        scaling factor. The choice of '4' as a default prevents the use of
        up-scaled faces in the refinement and output stages (whose input sizes
        are twice and four times that of the proposal stage).
    proposal_probability_threshold : float
    refine_probability_threshold : float
    output_probability_threshold : float
        The is-face probability threshold for a face to be accepted during each
        of the three filtering stages.
    proposal_maximum_iou : float
    refine_maximum_iou : float
    output_maximum_iou : float
        The threshold of the intersection-over-union above which two rectangles
        are duplicates during each of the three processing stages.
    """
    pyramid = ImagePyramid(image, min_size=12, downscale_factor=pyramid_downscale_factor)
    
    # First 'proposal' stage. Run convolution at multiple scales
    # ----------------------------------------------------------
    all_level_probs = []
    all_level_bboxes = []
    
    first_level = pyramid.closest_level(proposal_skip_downscale_factor)
    for level, image in enumerate(pyramid):
        if level < first_level:
            continue
        
        probs, bboxes = get_proposals(
            image_to_array(image),
            proposal_probability_threshold,
        )
        
        # Remove overlaps within level using NMS
        selection = non_maximum_suppression(probs, bboxes, proposal_maximum_iou)
        
        scale_to_native = pyramid.scale_between(level, 0)
        all_level_probs.append(probs[selection])
        all_level_bboxes.append(bboxes[selection] * scale_to_native)
    
    probs = np.concatenate(all_level_probs)
    bboxes = np.concatenate(all_level_bboxes)

    # Remove overlaps between propsals at different scales using NMS on
    # aggregated mappings
    selection = non_maximum_suppression(probs, bboxes, proposal_maximum_iou)
    probs = probs[selection]
    bboxes = bboxes[selection]

    # Second 'refinement' stage. Run network against each image in turn.
    # ------------------------------------------------------------------
    probs, bboxes = refine_proposals(pyramid, bboxes, refine_probability_threshold)
    
    # Remove overlapping candidates with NMS
    selection = non_maximum_suppression(probs, bboxes, refine_maximum_iou)
    probs = probs[selection]
    bboxes = bboxes[selection]

    # Third 'output' stage. Additional refinement and landmark location.
    # ------------------------------------------------------------------
    probs, bboxes, landmarks = output_proposals(pyramid, bboxes, output_probability_threshold)

    # Remove overlapping candidates with NMS
    selection = non_maximum_suppression(probs, bboxes, output_maximum_iou)
    probs = probs[selection]
    bboxes = bboxes[selection]
    landmarks = landmarks[selection]
    
    return DetectedFaces(probs, bboxes, landmarks)
