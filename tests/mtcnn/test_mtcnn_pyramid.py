import pytest

from PIL import Image

import numpy as np

from faceie.mtcnn.detect_faces import ImagePyramid


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

    def test_extract(self) -> None:
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
