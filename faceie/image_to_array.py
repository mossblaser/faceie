from PIL import Image

import numpy as np
from numpy.typing import NDArray


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
