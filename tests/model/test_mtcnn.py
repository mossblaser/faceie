import pytest

from PIL import Image

from pathlib import Path

import numpy as np

from faceie.model.mtcnn import (
    p_net,
    resolve_p_net_bounding_boxes,
    resolve_coordinates,
    non_maximum_suppression,
    make_square,
    ImagePyramid,
    DetectedFaces,
    detect_faces,
)


TEST_IMAGE_DIR = Path(__file__).parent


def test_p_net_finds_face() -> None:
    # This test case is just a sanity check that this finds the face in the
    # centre of the test image (which has been manually sized such that the
    # face is of an appropriate size for the detector).
    
    im = np.asarray(Image.open(TEST_IMAGE_DIR / "tiny_face.jpg"))
    im = np.moveaxis(im, 2, 0)  # (3, height, width)
    im = (im - 127.5) / 128.0
    
    probs, bboxes = p_net(im)
    bboxes = resolve_p_net_bounding_boxes(bboxes)
    
    # Find center of most likley face detection bounding box
    x1, y1, x2, y2 = bboxes.reshape(-1, 4)[np.argmax(probs)]
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    
    # Best face within 3 pixels of center of image
    assert np.isclose(x, im.shape[2] / 2, atol=3)
    assert np.isclose(y, im.shape[1] / 2, atol=3)


class TestNonMaximumSuppression:
    
    def test_empty(self) -> None:
        probs = np.array([])
        bboxes = np.array([]).reshape(0, 4)
        out = non_maximum_suppression(probs, bboxes, 0.5)
        assert out.shape == (0, )
    
    def test_singleton(self) -> None:
        probs = np.array([0.9])
        bboxes = np.array([[0, 1, 2, 3]])
        out = non_maximum_suppression(probs, bboxes, 0.5)
        assert out.shape == (1, )
        assert out[0] == 0
    
    def test_minimum_iou(self) -> None:
        # First and second overlap by exactly 50%, first and third by 25%;
        probs = np.array([1.0, 0.9, 0.8])
        bboxes = np.array([[0, 0, 8, 8], [2, 0, 10, 8], [4, 4, 12, 12]])
        out = non_maximum_suppression(probs, bboxes, 0.5)
        assert out.shape == (2, )
        assert set(out) == {0, 2}


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


class TestImagePyramid:
    
    @pytest.mark.parametrize("rotated", [False, True])
    def test_level_sizes(self, rotated: bool) -> None:
        w, h = 256, 128
        min_size = 12
        
        if rotated:
            w, h = h, w
        
        p = ImagePyramid(Image.new("RGB", (w, h)), min_size)
        
        assert len(p) == 4
        
        s = slice(None, None, -1) if rotated else slice(None)
        assert p[0].size[s] == (256, 128)
        assert p[1].size[s] == (128, 64)
        assert p[2].size[s] == (64, 32)
        assert p[3].size[s] == (32, 16)
    
    @pytest.mark.parametrize(
        "a, b, exp",
        [
            (0, 1, 0.5),
            (1, 0, 2.0),
            (1, 3, 0.25),
            (3, 1, 4),
        ],
    )
    def test_scale_between(self, a: int, b: int, exp: float) -> None:
        p = ImagePyramid(Image.new("RGB", (128, 64)), 12)
        assert p.scale_between(a, b) == exp
    
    @pytest.mark.parametrize(
        "downscale_factor, exp",
        [
            # Identical to input
            (1.0, 0),
            # Smaller than input, but still larger than half the resolution
            (1.5, 0),
            # Exactly half the input resolution
            (2.0, 1),
            # Over half, but not yet a quarter
            (3.0, 1),
            # A quarter
            (4.0, 2),
            # Smaller than the smallest level (should just use the smallest
            # level)
            (9999.0, 2),
            # Larger than the input (should just use the input)
            (0.5, 0),
        ],
    )
    def test_closest_level(self, downscale_factor: float, exp: int) -> None:
        p = ImagePyramid(Image.new("RGB", (128, 64)), 12)
        assert p.closest_level(downscale_factor) == exp
    
    def test_scale_between(self) -> None:
        pixels = np.zeros((128, 256, 3), dtype=np.uint8)
        pixels[24:, 48:] = 0xFF
        im = Image.fromarray(pixels, mode="RGB")
        
        p = ImagePyramid(im, 12)

        out = p.extract((32, 16, 64, 32), (10, 5))
        
        assert out.size == (10, 5)
        
        assert out.getpixel((0, 0)) == (0x00, 0x00, 0x00)
        assert out.getpixel((9, 0)) == (0x00, 0x00, 0x00)
        assert out.getpixel((0, 4)) == (0x00, 0x00, 0x00)
        assert out.getpixel((9, 4)) == (0xFF, 0xFF, 0xFF)
    
    def test_out_of_range(self) -> None:
        pixels = np.zeros((128, 256, 3), dtype=np.uint8)
        pixels[0, 0] = 0xFF
        im = Image.fromarray(pixels, mode="RGB")
        
        p = ImagePyramid(im, 12)

        out = p.extract((-16, -16, 16, 16))
        
        # Don't care about the values which are out-of-range but those in-range
        # must match the input.
        assert out.size == (32, 32)
        assert out.getpixel((16, 16)) == (0xFF, 0xFF, 0xFF)
        assert out.getpixel((17, 16)) == (0x00, 0x00, 0x00)
        assert out.getpixel((16, 17)) == (0x00, 0x00, 0x00)
        assert out.getpixel((17, 17)) == (0x00, 0x00, 0x00)


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
    
    assert lx == pytest.approx(1/5, abs=0.1)
    assert rx == pytest.approx(4/5, abs=0.1)
    
    assert ly == pytest.approx(2/5, abs=0.1)
    assert ry == pytest.approx(2/5, abs=0.1)
