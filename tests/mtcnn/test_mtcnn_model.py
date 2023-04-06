import pytest

from PIL import Image

from pathlib import Path

import numpy as np

from faceie.mtcnn.model import p_net
from faceie.mtcnn.detect_faces import resolve_p_net_bounding_boxes


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


