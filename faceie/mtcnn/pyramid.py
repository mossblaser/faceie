"""
An implementation of a simple image pyramid structure.

.. autoclass:: ImagePyramid
"""

from typing import Iterator

from PIL import Image

import math


class ImagePyramid:
    """
    Implements a simple image 'pyramid' containing multiple copies of an input
    image scaled down by a factor of two until a particular minimum size.
    """

    # The full set of images in the pyramid. The i-th image is a factor of 2^i
    # smaller than the input image.
    _images: list[Image.Image]

    def __init__(
        self, image: Image.Image, min_size: int, downscale_factor: float = 2.0
    ) -> None:
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
            max(
                0,
                math.floor(
                    math.log(downscale_factor) / math.log(self.downscale_factor)
                ),
            ),
        )

    def extract(
        self,
        crop: tuple[float, float, float, float],
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
        iw = round(x2 - x1)
        ih = round(y2 - y1)

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
