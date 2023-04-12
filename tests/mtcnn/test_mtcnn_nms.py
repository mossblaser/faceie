import numpy as np

from faceie.mtcnn.non_maximum_suppression import non_maximum_suppression


class TestNonMaximumSuppression:
    def test_empty(self) -> None:
        probs = np.array([])
        bboxes = np.array([]).reshape(0, 4)
        out = non_maximum_suppression(probs, bboxes, 0.5)
        assert out.shape == (0,)

    def test_singleton(self) -> None:
        probs = np.array([0.9])
        bboxes = np.array([[0, 1, 2, 3]])
        out = non_maximum_suppression(probs, bboxes, 0.5)
        assert out.shape == (1,)
        assert out[0] == 0

    def test_minimum_iou(self) -> None:
        # First and second overlap by exactly 50%, first and third by 25%;
        probs = np.array([1.0, 0.9, 0.8])
        bboxes = np.array([[0, 0, 8, 8], [2, 0, 10, 8], [4, 4, 12, 12]])
        out = non_maximum_suppression(probs, bboxes, 0.5)
        assert out.shape == (2,)
        assert set(out) == {0, 2}
