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
