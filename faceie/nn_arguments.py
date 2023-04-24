"""
Containers for storing the arguments to various functions in
:py:mod:`faceie.nn`.
"""


from typing import NamedTuple

from numpy.typing import NDArray


class LinearWeights(NamedTuple):
    """
    Linear layer weights, as used by :py:func:`faceie.nn.linear`.
    """

    weights: NDArray
    biases: NDArray | None


class Conv2DWeights(NamedTuple):
    """
    Convolutional kernel and biases for use by :py:func:`faceie.nn.conv2d`.
    """

    weights: NDArray
    biases: NDArray | None


class BatchNormalisationWeights(NamedTuple):
    """
    The mean, variance, weights and biases learnt during batch normalisation,
    as used by :py:func:`faceie.nn.batch_normalisation_2d`
    """

    population_mean: NDArray
    population_variance: NDArray

    weights: NDArray
    biases: NDArray

    eps: float
