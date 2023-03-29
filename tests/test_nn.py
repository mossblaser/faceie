import pytest

import numpy as np
from numpy.typing import NDArray

import torch

from faceie.nn import conv2d, prelu, max_pool_2d, softmax


def test_conv2d() -> None:
    # Compare behaviour against PyTorch
    
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
    assert torch_conv2d.weight.shape == (out_channels, in_channels, kernel_height, kernel_width)
    assert torch_conv2d.bias.shape == (out_channels, )
    
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


class TestMaxPool2D:
    
    @pytest.mark.parametrize("ceil_mode", [True, False])
    def test_perfect_fit_kernel_and_stride(self, ceil_mode: bool) -> None:
        ar = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 0, 1, 2],
                [3, 4, 5, 6, 7, 8],
                [9, 0, 1, 2, 3, 4],
            ],
        )
        
        assert np.array_equal(
            max_pool_2d(ar, (2, 3), (2, 3), ceil_mode=ceil_mode),
            np.array(
                [
                    [9, 6],
                    [9, 8],
                ]
            ),
        )
    
    def test_not_ceil_mode_truncates(self) -> None:
        ar = np.array(
            [
                [1, 2, 3, 4, 5],
                [7, 8, 9, 0, 1],
                [3, 4, 5, 6, 7],
            ],
        )
        
        assert np.array_equal(
            max_pool_2d(ar, (2, 3), (2, 3), ceil_mode=False),
            np.array(
                [
                    [9],
                ]
            ),
        )
    
    @pytest.mark.parametrize(
        "description, input, exp, kernel, stride",
        [
            # Square kernel
            (
                "Stride 1, square kernel",
                np.array(
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 1],
                    ],
                ),
                np.array(
                    [
                        [7, 8, 9, 9],
                    ]
                ),
                (2, 2),
                (1, 1),
            ),
            (
                "Stride 1, rectangular kernel",
                np.array(
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 1],
                    ],
                ),
                np.array(
                    [
                        [2, 3, 4, 5],
                        [7, 8, 9, 9],
                    ]
                ),
                (1, 2),
                (1, 1),
            ),
            (
                "Stride 2, kernel 3, sizes work out, no padding required",
                np.array(
                    [
                        [1, 2, 3, 4, 5],
                    ],
                ),
                np.array(
                    [
                        [3, 5],
                    ]
                ),
                (1, 3),
                (1, 2),
            ),
            (
                "Stride 2, kernel 3, sizes don't work out, padding required",
                np.array(
                    [
                        [1, 2, 3, 4, 5, 6],
                    ],
                ),
                np.array(
                    [
                        [3, 5, 6],
                    ]
                ),
                (1, 3),
                (1, 2),
            ),
            (
                "Stride 3, kernel 3, edges and corner padding required",
                np.array(
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 8],
                        [7, 6, 5, 4, 3],
                        [2, 1, 0, 1, 2],
                    ],
                ),
                np.array(
                    [
                        [8, 9],
                        [2, 2],
                    ]
                ),
                (3, 3),
                (3, 3),
            ),
        ]
    )
    @pytest.mark.parametrize("transpose", [False, True])
    @pytest.mark.parametrize("extra_dims", [False, True])
    def test_ceil_mode(
        self,
        description: str,
        input: NDArray,
        exp: NDArray,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        transpose: bool,
        extra_dims: bool,
    ) -> None:
        if transpose:
            input = np.swapaxes(input, -2, -1)
            exp = np.swapaxes(exp, -2, -1)
            kernel = kernel[::-1]
            stride = stride[::-1]
        
        if extra_dims:
            input = np.stack((input, input * 2))
            exp = np.stack((exp, exp * 2))
        
        print(f"{input=}")
        print(f"{kernel=}")
        print(f"{stride=}")
        out = max_pool_2d(input, kernel=kernel, stride=stride, ceil_mode=True)
        print(f"{out=}")
        print(f"{exp=}")
        assert np.array_equal(out, exp), description


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
