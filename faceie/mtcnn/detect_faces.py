"""
This submodule implements the overall face detection process
(:py:func:`detect_faces`).
"""


from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from PIL import Image

from faceie.mtcnn.pyramid import ImagePyramid
from faceie.mtcnn.model import p_net, r_net, o_net
from faceie.mtcnn.non_maximum_suppression import non_maximum_suppression



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
    """
    The output of :py:func:`detect_faces`.
    """
    
    probabilities: NDArray
    """
    A (num_faces, ) shaped array giving the probability score (0.0 - 1.0)
    assigned to each detected face.
    """
    
    bounding_boxes: NDArray
    """
    A (num_faces, 4) shaped array giving the coordinates of a bounding box for
    each detected face as x1, y1, x2 and y2. Coordinates are given in terms of
    input pixels and are *inclusive*.
    """
    
    landmarks: NDArray
    """
    A (num_faces, 10) shaped array giving the coordinates of five detected facial
    landmarks in each face. These are given as a series of pixel coordinates
    (x1, y1, x2, y2, ...) with the landmarks being:
    
    * Left eye
    * Right eye
    * Nose
    * Left of mouth
    * Right of mouth
    """


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
    
    Returns
    =======
    DetectedFaces
        For each detected face, a probability, bounding box and set of facial
        landmarks.
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
