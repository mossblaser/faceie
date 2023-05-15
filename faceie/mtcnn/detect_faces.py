"""
This submodule implements the overall face detection process
(:py:func:`detect_faces`).
"""


from typing import NamedTuple

from itertools import chain

import numpy as np
from numpy.typing import NDArray

from PIL import Image

from faceie.image_to_array import image_to_array
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


def bounding_boxes_to_centers(bounding_boxes: NDArray) -> NDArray:
    """
    Given an array of bounding boxes, return their center coordinates.

    Parameters
    ==========
    bounding_boxes : array (..., 4)

    Returns
    =======
    array (..., 2)
    """
    return np.sum(bounding_boxes.reshape(-1, 2, 2), axis=-2) / 2.0


def bounding_boxes_to_longest_sides(bounding_boxes: NDArray) -> NDArray:
    """
    Given an array of bounding boxes, return their longest sides.

    Parameters
    ==========
    bounding_boxes : array (..., 4)

    Returns
    =======
    array (...)
    """
    lengths = np.diff(bounding_boxes.reshape(-1, 2, 2), axis=-2).reshape(-1, 2)
    return np.max(lengths, axis=-1)


def centers_to_square_bounding_boxes(centres: NDArray, sizes: NDArray) -> NDArray:
    """
    Given an array of suqare bounding box centers and their sizes, returns an
    array of bounding boxes.

    Parameters
    ==========
    centers: array (..., 2)
    sizes: array (...)

    Returns
    =======
    array (..., 4)
    """
    half_sizes = np.expand_dims(sizes / 2.0, -1)
    return np.concatenate(
        [
            centres - half_sizes,
            centres + half_sizes,
        ],
        axis=-1,
    )


def make_square(bounding_boxes: NDArray) -> NDArray:
    """
    Given an array of bounding boxes, enlarge these as necessary, centered on
    the existing centre point, to make them square.

    Parameters
    ==========
    bounding_boxes: array (..., 4)
        An array which contains a 4-long dimension enumerating the x1, y1, x2
        and y2 coordinates of a bounding box.

    Returns
    =======
    array (..., 4)
        A new bounding box array where all bounding boxes have x2-x1 == y2-y1.
    """
    centers = bounding_boxes_to_centers(bounding_boxes)
    longest_sides = bounding_boxes_to_longest_sides(bounding_boxes)
    return centers_to_square_bounding_boxes(centers, longest_sides)


def resolve_coordinates(
    input_bounding_boxes: NDArray, coordinate_pairs: NDArray
) -> NDArray:
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
    bounding box using the :py:func:`r_net`.

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
    crops = make_square(bounding_boxes)

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
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Given a series of potential faces, return a final (output) probability,
    bounding box and (optional) set of facial landmarks using the
    :py:func:`o_net`.

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
    crops = make_square(bounding_boxes)

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


def detect_faces_single_orientation(
    pyramid: ImagePyramid,
    proposal_skip_downscale_factor: float = 4.0,
    proposal_probability_threshold: float = 0.7,
    refine_probability_threshold: float = 0.7,
    output_probability_threshold: float = 0.7,
    proposal_maximum_iou: float = 0.5,
    refine_maximum_iou: float = 0.5,
    output_maximum_iou: float = 0.5,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Detect faces within an image using the MTCNN algorithm by Zhang et al.

    Parameters
    ==========
    pyramid : ImagePyramid
        An Image Pyramid with the image to detect faces within.
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
    # First 'proposal' stage. Run convolution at multiple scales
    # ----------------------------------------------------------
    all_level_probs = []
    all_level_bboxes = []

    first_level = pyramid.closest_level(proposal_skip_downscale_factor)
    for level, image in enumerate(pyramid):
        if level < first_level or image.size[0] < 12 or image.size[1] < 12:
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
    probs, bboxes, landmarks = output_proposals(
        pyramid, bboxes, output_probability_threshold
    )

    # Remove overlapping candidates with NMS
    selection = non_maximum_suppression(probs, bboxes, output_maximum_iou)
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

    angles: NDArray
    """
    A (num_faces, ) shaped array giving the angle at which the image was
    rotated (in radians) when a given face was detected.
    """

    bounding_boxes: NDArray
    """
    A (num_faces, 4) shaped array giving the coordinates of a bounding box for
    each detected face as x1, y1, x2 and y2. Coordinates are given in terms of
    input pixels and are *inclusive*.
    
    Note that the coordinates given are the un-rotated coordinates (i.e. those
    in the un-rotated input image). When angle != 0, you will need to rotate
    the image (and these coordinates) to extract a face accordingly.
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

    images: list[Image.Image] | None
    """
    If requested, includes a cropped image of each face (rotated to the angle
    at which it was recognised by the model).
    
    All images are square-cropped by expanding the bounding box area into a
    square. To obtain just the bounding-box area, you will need to further crop
    these images according to the aspect ratio of the corresponding bounding
    box.
    """


def translate(coords: NDArray, dx: float, dy: float) -> NDArray:
    """
    Apply a translation to a set of coordinates.

    Parameters
    ==========
    coords : array (..., even-number-of-coords)
        The Input coordintaes in the final dimension (where even numbered
        entries are x-coordinates and odd numbered entries are y-coordinates.
    dx : float
    dy : float
        The translation to be applied
    """
    out = coords.copy()
    out.T[0::2] += dx
    out.T[1::2] += dy
    return out


def rotate(coords: NDArray, angle: float) -> NDArray:
    """
    Rotate a collection of coordinates around the origin, counter-clockwise.

    Parameters
    ==========
    coords : array (..., even-number-of-coords)
        The Input coordintaes in the final dimension (where even numbered
        entries are x-coordinates and odd numbered entries are y-coordinates.
    angle : float
        Angle (in radians) to rotate the coordinates by.
    """
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    # (..., 2N) -> (..., N, 2)
    coords = coords.reshape(*coords.shape[:-1], coords.shape[-1] // 2, 2)
    # (..., N, 2) -> (..., 2, N)
    coords = np.moveaxis(coords, -1, -2)

    # Perform the actual rotation
    out = rotation @ coords  # (..., 2, N)

    # (..., 2, N) -> (..., N, 2)
    out = np.moveaxis(out, -1, -2)
    # (..., N, 2) -> (..., 2N)
    out = out.reshape(*out.shape[:-2], out.shape[-2] * 2)

    return out


def detect_faces(
    image: Image.Image,
    pyramid_downscale_factor: float = np.sqrt(2),
    rotations: NDArray = np.array([-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2]),
    proposal_skip_downscale_factor: float = 4.0,
    proposal_probability_threshold: float = 0.7,
    refine_probability_threshold: float = 0.7,
    output_probability_threshold: float = 0.7,
    proposal_maximum_iou: float = 0.5,
    refine_maximum_iou: float = 0.5,
    output_maximum_iou: float = 0.5,
    rotation_maximum_iou: float = 0.25,
    extract_images: int | None = 160,
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
    rotations : [radians, ...]
        The angles (in radians) at which to rotate the input image when searching.
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
    rotation_maximum_iou : float
        The threshold of the intersection-over-union above which two rectangles
        are duplicates during each of the three processing stages and the final
        combining of rotated detections.
    extract_images : int or None
        If a number of pixels is given, includes a
        extract_images-by-extract_images pixel square-crop of each detected
        face in the output. If None, face crops are not extracted.

        The default size of 160x160 pixels is used by the FaceNet-PyTorch
        implementation of the FaceNet model. This differs from the original
        FaceNet/Inception-ResNet-v1 paper which is shown using an input size of
        299x299 pixels. We use the former size by default to better match
        FaceNet-PyTorch from which we're re-using the weights.

    Returns
    =======
    DetectedFaces
        For each detected face, a probability, angle, bounding box, set of
        facial landmarks and (optionally) a cropped image.
    """
    all_pyramids = []
    all_probs = []
    all_angles = []
    all_bboxes = []
    all_bboxes_normalised = (
        []
    )  # Internal use: Corners of a square centered on the detected face
    all_bboxes_rotated = []  # Internal use: Bounding box in rotated image coordinates
    all_landmarks = []

    # Run face detection on all rotations
    for angle_rad in rotations:
        angle_deg = (angle_rad / np.pi) * 180
        if angle_rad != 0:
            rotated_image = image.rotate(
                angle_deg, resample=Image.Resampling.BICUBIC, expand=True
            )
        else:
            rotated_image = image

        pyramid = ImagePyramid(
            rotated_image,
            min_size=12,
            downscale_factor=pyramid_downscale_factor,
        )

        probs, bboxes, landmarks = detect_faces_single_orientation(
            pyramid=pyramid,
            proposal_skip_downscale_factor=proposal_skip_downscale_factor,
            proposal_probability_threshold=proposal_probability_threshold,
            refine_probability_threshold=refine_probability_threshold,
            output_probability_threshold=output_probability_threshold,
            proposal_maximum_iou=proposal_maximum_iou,
            refine_maximum_iou=refine_maximum_iou,
            output_maximum_iou=output_maximum_iou,
        )

        # NB: bounding-box and landmark coordinates rotated into original image
        # coordinates.
        all_pyramids.append([pyramid] * len(probs))
        all_angles.append(np.full(probs.shape, angle_rad))
        all_probs.append(probs)
        all_bboxes.append(
            translate(
                rotate(
                    translate(
                        bboxes,
                        -rotated_image.size[0] / 2,
                        -rotated_image.size[1] / 2,
                    ),
                    angle_rad,
                ),
                image.size[0] / 2,
                image.size[1] / 2,
            )
        )
        all_landmarks.append(
            translate(
                rotate(
                    translate(
                        landmarks,
                        -rotated_image.size[0] / 2,
                        -rotated_image.size[1] / 2,
                    ),
                    angle_rad,
                ),
                image.size[0] / 2,
                image.size[1] / 2,
            )
        )

        # For the purposes of later NMS, we normalise the detected bounding
        # boxes to squares oriented at 0 degrees centered on their original
        # location. This isn't ideal but it is easy!
        longest_sides = bounding_boxes_to_longest_sides(bboxes)
        centers = translate(
            rotate(
                translate(
                    bounding_boxes_to_centers(bboxes),
                    -rotated_image.size[0] / 2,
                    -rotated_image.size[1] / 2,
                ),
                angle_rad,
            ),
            image.size[0] / 2,
            image.size[1] / 2,
        )
        all_bboxes_normalised.append(
            centers_to_square_bounding_boxes(centers, longest_sides)
        )

        # Bounding-box coordinates in rotated image domain (used to extract
        # face crops)
        all_bboxes_rotated.append(bboxes)

    # Combine results
    pyramids = list(chain(*all_pyramids))
    probs = np.concatenate(all_probs)
    angles = np.concatenate(all_angles)
    bboxes = np.concatenate(all_bboxes)
    bboxes_normalised = np.concatenate(all_bboxes_normalised)
    bboxes_rotated = np.concatenate(all_bboxes_rotated)
    landmarks = np.concatenate(all_landmarks)

    # Remove duplicates
    selection = non_maximum_suppression(probs, bboxes_normalised, rotation_maximum_iou)

    pyramids = [pyramids[i] for i in selection]
    probs = probs[selection]
    angles = angles[selection]
    bboxes = bboxes[selection]
    bboxes_rotated = bboxes_rotated[selection]
    landmarks = landmarks[selection]

    # Extract face images from input
    images: list[Image.Image] | None = None
    if extract_images is not None:
        images = [
            pyramid.extract((x1, y1, x2, y2), (extract_images, extract_images))
            for pyramid, (x1, y1, x2, y2) in zip(pyramids, make_square(bboxes_rotated))
        ]

    return DetectedFaces(probs, angles, bboxes, landmarks, images)
