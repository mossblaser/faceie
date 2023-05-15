import pytest

from PIL import Image

from pathlib import Path

import numpy as np

import torch

from faceie.image_to_array import image_to_array
from faceie.mtcnn.model import p_net, r_net, o_net
from faceie.mtcnn.detect_faces import resolve_p_net_bounding_boxes

import facenet_pytorch.models.mtcnn  # type: ignore


TEST_IMAGE_DIR = Path(__file__).parent


def test_p_net_finds_face() -> None:
    # This test case is just a sanity check that this finds the face in the
    # centre of the test image (which has been manually sized such that the
    # face is of an appropriate size for the detector).
    im = image_to_array(Image.open(TEST_IMAGE_DIR / "tiny_face.jpg"))

    probs, bboxes = p_net(im)
    bboxes = resolve_p_net_bounding_boxes(bboxes)

    # Find center of most likley face detection bounding box
    x1, y1, x2, y2 = bboxes.reshape(-1, 4)[np.argmax(probs)]
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0

    # Best face within 3 pixels of center of image
    assert np.isclose(x, im.shape[2] / 2, atol=3)
    assert np.isclose(y, im.shape[1] / 2, atol=3)


def test_p_net_equivalent() -> None:
    im = image_to_array(Image.open(TEST_IMAGE_DIR / "tiny_face.jpg"))

    torch_p_net = facenet_pytorch.models.mtcnn.PNet()

    with torch.no_grad():
        exp_bboxes, exp_probs = torch_p_net(torch.tensor(im).unsqueeze(0))

    probs, bboxes = p_net(im)

    assert np.allclose(probs, exp_probs.numpy()[:, 1])
    assert np.allclose(
        bboxes,
        np.moveaxis(
            (
                exp_bboxes.numpy()  # (1, 4, h, w)
                + np.array([0, 0, 1, 1]).reshape(1, 4, 1, 1)
            ),
            1,
            -1,
        ),  # (1, h, w, 4)
        atol=1e-5,  # Float32 is a bit naff
    )


def test_r_net_equivalent() -> None:
    im = image_to_array(Image.open(TEST_IMAGE_DIR / "tiny_face.jpg").resize((24, 24)))
    im = np.expand_dims(im, 0)

    torch_r_net = facenet_pytorch.models.mtcnn.RNet()

    with torch.no_grad():
        exp_bboxes, exp_probs = torch_r_net(torch.tensor(im))

    probs, bboxes = r_net(im)

    assert np.allclose(probs, exp_probs.numpy()[:, 1])
    assert np.allclose(
        bboxes,
        (exp_bboxes.numpy() + np.array([0, 0, 1, 1]).reshape(1, 4)),  # (1, 4)
        atol=1e-5,  # Float32 is a bit naff
    )


def test_o_net_equivalent() -> None:
    im = image_to_array(Image.open(TEST_IMAGE_DIR / "tiny_face.jpg").resize((48, 48)))
    im = np.expand_dims(im, 0)

    torch_o_net = facenet_pytorch.models.mtcnn.ONet()

    with torch.no_grad():
        exp_bboxes, exp_landmarks, exp_probs = torch_o_net(torch.tensor(im))

    probs, bboxes, landmarks = o_net(im)

    assert np.allclose(probs, exp_probs.numpy()[:, 1])

    assert np.allclose(
        bboxes,
        (exp_bboxes.numpy() + np.array([0, 0, 1, 1]).reshape(1, 4)),  # (1, 4)
        atol=1e-5,  # Float32 is a bit naff
    )

    assert np.allclose(
        landmarks,
        (exp_landmarks.numpy()[:, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]),
        atol=1e-5,  # Float32 is a bit naff
    )
