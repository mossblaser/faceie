import pytest

import numpy as np
from numpy.typing import NDArray

import torch

from faceie.nn import (
    l2_normalisation,
    PaddingType,
    batch_normalisation_2d,
    conv2d,
    relu,
    prelu,
    max_pool_2d,
    softmax,
)


def test_l2_normalisation() -> None:
    rand = np.random.RandomState(0)

    x = rand.uniform(-10, 10, size=(3, 4, 5)).astype(np.float32)

    out = l2_normalisation(x, axis=1)

    # Work out model answer using PyTorch implementation
    exp_tensor = torch.nn.functional.normalize(torch.tensor(x), p=2.0, dim=1)
    exp = exp_tensor.numpy()

    print(out)
    print(exp)

    print(np.sum(out**2, axis=1))
    print(np.sum(exp**2, axis=1))

    assert np.allclose(out, exp, atol=1e-6)


@pytest.mark.parametrize("channel_axis", [1, -3])
def test_batch_normalisation_2d(channel_axis: int) -> None:
    rand = np.random.RandomState(0)

    num_batches = 2
    num_channels = 3

    x = rand.uniform(-10, 10, size=(num_batches, num_channels, 2, 2)).astype(np.float32)

    population_mean = rand.normal(size=num_channels).astype(np.float32)
    population_variance = np.abs(rand.normal(size=num_channels).astype(np.float32))
    weights = rand.normal(size=num_channels).astype(np.float32)
    biases = rand.normal(size=num_channels).astype(np.float32)
    eps = 1e-5

    out = batch_normalisation_2d(
        x,
        population_mean,
        population_variance,
        weights,
        biases,
        eps=eps,
        channel_axis=channel_axis,
    )

    # Work out model answer using PyTorch implementation
    torch_batch_normalisation_2d = torch.nn.BatchNorm2d(num_channels, eps=eps).eval()
    with torch.no_grad():
        assert torch_batch_normalisation_2d.running_mean is not None
        assert torch_batch_normalisation_2d.running_var is not None
        torch_batch_normalisation_2d.running_mean[:] = torch.tensor(population_mean)
        torch_batch_normalisation_2d.running_var[:] = torch.tensor(population_variance)
        torch_batch_normalisation_2d.weight[:] = torch.tensor(weights)
        torch_batch_normalisation_2d.bias[:] = torch.tensor(biases)

        exp = torch_batch_normalisation_2d(torch.tensor(x)).numpy()

    print(out)
    print(exp)

    assert np.allclose(out, exp, atol=1e-6)


class TestConv2D:
    @pytest.mark.parametrize(
        "kernel, stride, padding",
        [
            # No stride or padding
            ((3, 3), 1, 0),
            # Not square
            ((3, 5), 1, 0),
            # Strided
            ((3, 3), 2, 0),
            ((3, 3), (2, 1), 0),
            ((3, 3), (1, 2), 0),
            ((3, 5), (1, 2), 0),  # Also not square!
            # Padded
            ((3, 3), 1, 1),
            ((5, 5), 1, 2),
            ((5, 5), 1, (2, 0)),
            ((5, 5), 1, (0, 2)),
            ((3, 5), 1, (0, 2)),  # Also not square!
        ],
    )
    def test_one_hot(
        self,
        kernel: tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
    ) -> None:
        # This test convolves the input with various kernel sizes, strides and
        # paddings when processing a series of one-hot input arrays. These
        # provide an easy-to-debug test case for working out issues with
        # padding and suchlike.

        weights = np.arange(1, (kernel[0] * kernel[1]) + 1, dtype=np.float32).reshape(
            1, 1, *kernel
        )
        biases = np.array([100], dtype=np.float32)

        height = 6
        width = 10

        # Compare output with PyTorch implementation
        torch_conv2d = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        with torch.no_grad():
            torch_conv2d.weight[:] = torch.tensor(weights)
            assert torch_conv2d.bias is not None
            torch_conv2d.bias[:] = torch.tensor(biases)

        for i in range(width * height):
            img = np.zeros((1, 1, height, width), dtype=np.float32)
            img.flat[i] = 1

            print("")
            print(f"{height=}, {width=}, {kernel=}, {stride=}, {padding=}")
            print(f"{img.shape=}")
            print(img)

            out = conv2d(img, weights, biases, stride=stride, padding=padding)

            print(f"{out.shape=}")
            print(out)

            with torch.no_grad():
                exp_out = torch_conv2d(torch.tensor(img)).numpy()

            print(f"{exp_out.shape=}")
            print(exp_out)

            assert np.array_equal(out, exp_out)

    def test_batches_and_channels(self) -> None:
        # Compare behaviour against PyTorch with a testcase with random weights
        # and inputs which uses batching and  multiple input and output
        # channels.
        num_batches = 2
        in_channels = 3
        out_channels = 4
        kernel_height = 5
        kernel_width = 7

        img_height = 768
        img_width = 1024

        torch_conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_height, kernel_width),
        )

        # Sanity check
        assert torch_conv2d.weight is not None
        assert torch_conv2d.bias is not None
        assert torch_conv2d.weight.shape == (
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        )
        assert torch_conv2d.bias.shape == (out_channels,)

        # Get model answer
        img_tensor = torch.randn(num_batches, in_channels, img_height, img_width)
        with torch.no_grad():
            torch_out = torch_conv2d(img_tensor)

        # Convert image/weights/biases to NDArray
        img = img_tensor.numpy()
        weights = torch_conv2d.weight.detach().numpy()
        biases = torch_conv2d.bias.detach().numpy()

        out = conv2d(img, weights, biases)

        # Sanity check shape
        out_height = img_height - ((kernel_height // 2) * 2)
        out_width = img_width - ((kernel_width // 2) * 2)
        assert out.shape == (num_batches, out_channels, out_height, out_width)

        # Check equivalent to PyTorch
        #
        # NB higher tollerance due to float32 precision
        assert np.allclose(out, torch_out.numpy(), atol=1e-6)

    def test_padding_same(self) -> None:
        kernel = (5, 3)

        weights = np.zeros((1, 1, kernel[0], kernel[1]), dtype=np.float32)

        height = 6
        width = 10

        img = np.zeros((1, height, width), dtype=np.float32)

        out = conv2d(img, weights, padding=PaddingType.same)

        assert out.shape == img.shape


def test_relu() -> None:
    num_channels = 10

    input_shape = (3, num_channels, 100, 200)

    torch_relu = torch.nn.ReLU(inplace=False)

    # Get model answer
    in_tensor = torch.randn(*input_shape)
    with torch.no_grad():
        out_tensor = torch_relu(in_tensor)

    # Convert types
    input = in_tensor.numpy()

    out = relu(input)

    assert np.allclose(out, out_tensor.numpy(), atol=1e-6)


def test_prelu() -> None:
    num_channels = 10

    # Arbitrary choice except for the axis of choice being at index 1 which is
    # assumed by PyTorch.
    input_shape = (3, num_channels, 100, 200)

    torch_prelu = torch.nn.PReLU(num_channels)
    with torch.no_grad():
        # Randomise weights
        torch_prelu.weight[:] = torch.rand(*torch_prelu.weight.shape) * 2

    # Get model answer
    in_tensor = torch.randn(*input_shape)
    with torch.no_grad():
        out_tensor = torch_prelu(in_tensor)

    # Convert types
    input = in_tensor.numpy()
    parameters = torch_prelu.weight.detach().numpy()

    out = prelu(input, parameters, axis=1)

    assert np.allclose(out, out_tensor.numpy(), atol=1e-6)


@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("square", [False, True])
def test_max_pool_2d(ceil_mode: bool, square: bool) -> None:
    for stride in range(1, 5):
        for kernel in range(stride, 5):

            torch_max_pool_2d = torch.nn.MaxPool2d(
                kernel_size=kernel if square else (kernel, kernel + 1),
                stride=stride if square else (stride, stride + 1),
                ceil_mode=ceil_mode,
            )

            # Work through all a one-hot 2D arrays of the following size
            w = 8
            h = 6
            for i in range(w * h):
                ar = np.zeros(w * h, dtype=float).reshape(1, 1, h, w)
                ar.flat[i] = 1.0

                out = max_pool_2d(
                    ar,
                    kernel=kernel if square else (kernel, kernel + 1),
                    stride=stride if square else (stride, stride + 1),
                    ceil_mode=ceil_mode,
                )
                with torch.no_grad():
                    exp_out = torch_max_pool_2d(torch.tensor(ar)).numpy()

                assert np.array_equal(out, exp_out), f"{out=}\n{exp_out=}"


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_softmax(dim: int) -> None:
    np.random.seed(0)
    x = np.random.uniform(-10, 10, size=(3, 3))

    torch_softmax = torch.nn.Softmax(dim)
    exp = torch_softmax(torch.tensor(x)).detach().numpy()

    actual = softmax(x, axis=dim)

    print(exp)
    print(actual)

    assert np.allclose(actual, exp)
