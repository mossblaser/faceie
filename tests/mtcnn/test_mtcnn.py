import pytest

from PIL import Image

from pathlib import Path

import numpy as np

from faceie.mtcnn.detect_faces import (
    make_square,
    ImagePyramid,
    DetectedFaces,
    detect_faces,
)


TEST_IMAGE_DIR = Path(__file__).parent


@pytest.mark.parametrize("axis", [2, -1])
def test_make_square(axis: int) -> None:
    input = np.array(
        [
            [
                [10, 10, 20, 20],  # Already square
                [10, 10, 20, 16],  # Wide
                [10, 10, 16, 20],  # Tall
            ]
        ]
    )

    exp = np.array(
        [
            [
                [10, 10, 20, 20],
                [10, 8, 20, 18],
                [8, 10, 18, 20],
            ]
        ]
    )

    out = make_square(input, axis)

    assert np.array_equal(out, exp)
    assert out.dtype is input.dtype


def test_detect_faces() -> None:
    # Just sanity check that it finds the two faces in this image (cropped
    # such that they are at approximatley the right place.
    im = Image.open(TEST_IMAGE_DIR / "two_faces.jpg")
    probs, bboxes, landmarks = detect_faces(im)

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
