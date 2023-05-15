import pytest

from PIL import Image

from pathlib import Path

import numpy as np

from faceie.mtcnn.detect_faces import (
    bounding_boxes_to_centers,
    bounding_boxes_to_longest_sides,
    centers_to_square_bounding_boxes,
    make_square,
    ImagePyramid,
    DetectedFaces,
    detect_faces_single_orientation,
    detect_faces,
    translate,
    rotate,
)


TEST_IMAGE_DIR = Path(__file__).parent


def test_bounding_boxes_to_centers() -> None:
    ar = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    assert np.array_equal(
        bounding_boxes_to_centers(ar),
        np.array(
            [
                [2, 3],
                [6, 7],
            ]
        ),
    )


def test_bounding_boxes_to_longest_sides() -> None:
    ar = np.array(
        [
            [1, 2, 4, 4],
            [5, 6, 8, 10],
        ]
    )

    assert np.array_equal(
        bounding_boxes_to_longest_sides(ar),
        np.array([3, 4]),
    )


def test_centers_to_square_bounding_boxes() -> None:
    ar = np.array(
        [
            [2, 3],
            [6, 7],
        ]
    )

    assert np.array_equal(
        centers_to_square_bounding_boxes(ar, np.array(2)),
        np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]
        ),
    )


def test_make_square() -> None:
    input = np.array(
        [
            [
                [10, 10, 20, 20],  # Already square
                [10, 10, 20, 16],  # Wide
                [10, 10, 16, 20],  # Tall
            ]
        ],
    )

    exp = np.array(
        [
            [
                [10, 10, 20, 20],
                [10, 8, 20, 18],
                [8, 10, 18, 20],
            ]
        ],
    )

    out = make_square(input)

    assert np.allclose(out, exp)


def test_translate() -> None:
    ar = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )

    assert np.array_equal(
        translate(ar, 100, 1000),
        np.array(
            [
                [101, 1002, 103, 1004],
                [105, 1006, 107, 1008],
                [109, 1010, 111, 1012],
                [113, 1014, 115, 1016],
            ]
        ),
    )


def test_rotate() -> None:
    ar = np.array(
        [
            [2, 0, 0, 2],
            [np.sqrt(2), np.sqrt(2), np.sqrt(2), -np.sqrt(2)],
        ]
    )

    assert np.allclose(
        rotate(ar, np.pi / 4),
        np.array(
            [
                [np.sqrt(2), np.sqrt(2), -np.sqrt(2), np.sqrt(2)],
                [0, 2, 2, 0],
            ]
        ),
    )


def test_detect_faces_single_orientation() -> None:
    # Just sanity check that it finds the two faces in this image (cropped
    # such that they are at approximatley the right place.
    im = Image.open(TEST_IMAGE_DIR / "two_faces.jpg")
    pyramid = ImagePyramid(im, 1)
    probs, bboxes, landmarks = detect_faces_single_orientation(pyramid)

    # Should have found two faces
    assert len(probs) == 2
    assert np.all(probs > 0.9)

    print(bboxes)
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
    centers[:, 0] /= im.size[0]
    centers[:, 1] /= im.size[1]

    # Check the face positions match where they were cropped/rotated into the
    # photo.
    (lx, ly), (rx, ry) = sorted(map(tuple, centers))

    assert lx == pytest.approx(1 / 5, abs=0.1)
    assert rx == pytest.approx(4 / 5, abs=0.1)

    assert ly == pytest.approx(2 / 5, abs=0.1)
    assert ry == pytest.approx(2 / 5, abs=0.1)


@pytest.mark.parametrize("extract_images", [160, None])
def test_detect_faces_multiple_orientations(extract_images: int | None) -> None:
    # Just sanity check that it finds the two (differently oriented) faces in this image
    im = Image.open(TEST_IMAGE_DIR / "faces_at_angles.jpg")
    out = detect_faces(
        im,
        rotations=np.array([-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2]),
        extract_images=extract_images,
    )

    # Should have found two faces
    assert len(out.probabilities) == 2

    # Should have found one upright face and one leaning to the right
    assert set(out.angles) == {0.0, np.pi / 4}

    leaning_face = np.argmax(out.angles)
    upright_face = np.argmin(out.angles)

    # The leaning face should be on the bottom-left third
    centers = bounding_boxes_to_centers(out.bounding_boxes)
    centers[..., 0] /= im.size[0]
    centers[..., 1] /= im.size[1]
    assert np.allclose(centers[leaning_face], [0.33, 0.66], atol=0.1)
    assert np.allclose(centers[upright_face], [0.66, 0.33], atol=0.1)

    # Landmarks should be within their bounding boxes (i.e. check at least the
    # rotations are consistent
    for angle, bbox, landmarks in zip(out.angles, out.bounding_boxes, out.landmarks):
        bbox = rotate(bbox, -angle)
        landmarks = rotate(landmarks, -angle)

        assert bbox[0] < bbox[2]
        assert bbox[1] < bbox[3]

        for x, y in landmarks.reshape(-1, 2):
            assert bbox[0] < x < bbox[2]
            assert bbox[1] < y < bbox[3]

    # Extracted images should be of right size (or absent, as required)
    if extract_images is None:
        assert out.images is None
    else:
        assert out.images is not None
        for image in out.images:
            assert image.size == (extract_images, extract_images)
