import pytest

import numpy as np

import torch

from faceie.nn import conv2d


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
