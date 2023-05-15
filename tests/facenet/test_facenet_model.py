import pytest

from pathlib import Path

from PIL import Image

import torch

import numpy as np
from numpy.typing import NDArray

from faceie.image_to_array import image_to_array
from faceie.scripts.convert_facenet_weights import extract_weights
from faceie.facenet.model import FaceNetWeights, encode_face

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


TEST_IMAGE_DIR = Path(__file__).parent


@pytest.fixture(scope="module")
def facenet_pytorch_model() -> InceptionResnetV1:
    # NB: Will trigger a modestly large download the first time it is called
    return InceptionResnetV1("vggface2").eval()


@pytest.fixture(scope="module")
def weights(facenet_pytorch_model: InceptionResnetV1) -> FaceNetWeights:
    return extract_weights(facenet_pytorch_model)


def test_same_output(
    facenet_pytorch_model: InceptionResnetV1, weights: FaceNetWeights
) -> None:
    image = Image.open(TEST_IMAGE_DIR / "jonathan_1.jpg")

    out = encode_face(image, weights)
    exp = facenet_pytorch_model(torch.tensor(image_to_array(image)).unsqueeze(0)).detach().numpy()

    assert np.allclose(out, exp, atol=1e-5)


def test_appears_to_work(weights: FaceNetWeights) -> None:
    images = [
        Image.open(TEST_IMAGE_DIR / name)
        for name in [
            "jonathan_1.jpg",
            "jonathan_2.jpg",
            "dara_1.jpg",
            "dara_2.jpg",
        ]
    ]

    encodings = encode_face(images, weights)

    for a in range(len(images)):
        person_a = a // 2
        for b in range(len(images)):
            person_b = b // 2

            score = np.sqrt(np.sum((encodings[a] - encodings[b]) ** 2))
            if person_a == person_b:
                assert score < 1.0
            else:
                assert score > 1.0
