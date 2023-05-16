"""
Convert the FaceNet weights from FaceNet-PyTorch's format into the structure used
by this library.
"""

from typing import Any

from argparse import ArgumentParser

import torch

import numpy as np
from numpy.typing import NDArray

from itertools import count

import weightie

from faceie.nn_arguments import (
    LinearWeights,
    Conv2DWeights,
    BatchNormalisationWeights,
)

from faceie.facenet.model import (
    FaceNetWeights,
    StemWeights,
    ConvolutionalUnitWeights,
    InceptionResNetAWeights,
    ReductionAWeights,
    InceptionResNetBWeights,
    ReductionBWeights,
    InceptionResNetCWeights,
)


def n(tensor: torch.Tensor) -> NDArray:
    """
    Convert a PyTorch tensor into a numpy array.
    """
    return tensor.detach().numpy().astype(np.float32)


def extract_convolutional_unit_weights(
    unit: Any,  # facenet_pytorch.models.inception_resnet_v1.BasicConv2d
) -> ConvolutionalUnitWeights:
    return ConvolutionalUnitWeights(
        kernel=n(unit.conv.weight),
        batch_normalisation=BatchNormalisationWeights(
            population_mean=n(unit.bn.running_mean),
            population_variance=n(unit.bn.running_var),
            weights=n(unit.bn.weight),
            biases=n(unit.bn.bias),
            # Hard-coded in FaceNet-PyTorch to match TensorFlow but not stored
            # in the serialised state_dict.
            eps=0.001,
        ),
    )


def extract_convolution_weights(unit: torch.nn.Conv2d) -> Conv2DWeights:
    assert unit.bias is not None
    return Conv2DWeights(
        weights=n(unit.weight),
        biases=n(unit.bias),
    )


def extract_weights(
    model: Any,  # facenet_pytorch.models.inception_resnet_v1.InceptionResnetV1
) -> FaceNetWeights:
    return FaceNetWeights(
        stem=StemWeights(
            extract_convolutional_unit_weights(model.conv2d_1a),
            extract_convolutional_unit_weights(model.conv2d_2a),
            extract_convolutional_unit_weights(model.conv2d_2b),
            extract_convolutional_unit_weights(model.conv2d_3b),
            extract_convolutional_unit_weights(model.conv2d_4a),
            extract_convolutional_unit_weights(model.conv2d_4b),
        ),
        inception_resnet_a=[
            InceptionResNetAWeights(
                branch_1_conv=extract_convolutional_unit_weights(module.branch0),
                branch_2_conv_1=extract_convolutional_unit_weights(module.branch1[0]),
                branch_2_conv_2=extract_convolutional_unit_weights(module.branch1[1]),
                branch_3_conv_1=extract_convolutional_unit_weights(module.branch2[0]),
                branch_3_conv_2=extract_convolutional_unit_weights(module.branch2[1]),
                branch_3_conv_3=extract_convolutional_unit_weights(module.branch2[2]),
                output_conv=extract_convolution_weights(module.conv2d),
                # Not stored in the serialised state_dict (but defined in
                # FaceNet-PyTorch)
                residual_scale=0.17,
            )
            for module in model.repeat_1
        ],
        reduction_a=ReductionAWeights(
            branch_1_conv=extract_convolutional_unit_weights(model.mixed_6a.branch0),
            branch_2_conv_1=extract_convolutional_unit_weights(
                model.mixed_6a.branch1[0]
            ),
            branch_2_conv_2=extract_convolutional_unit_weights(
                model.mixed_6a.branch1[1]
            ),
            branch_2_conv_3=extract_convolutional_unit_weights(
                model.mixed_6a.branch1[2]
            ),
        ),
        inception_resnet_b=[
            InceptionResNetBWeights(
                branch_1_conv=extract_convolutional_unit_weights(module.branch0),
                branch_2_conv_1=extract_convolutional_unit_weights(module.branch1[0]),
                branch_2_conv_2=extract_convolutional_unit_weights(module.branch1[1]),
                branch_2_conv_3=extract_convolutional_unit_weights(module.branch1[2]),
                output_conv=extract_convolution_weights(module.conv2d),
                # Not stored in the serialised state_dict (but defined in
                # FaceNet-PyTorch)
                residual_scale=0.10,
            )
            for module in model.repeat_2
        ],
        reduction_b=ReductionBWeights(
            branch_1_conv_1=extract_convolutional_unit_weights(
                model.mixed_7a.branch0[0]
            ),
            branch_1_conv_2=extract_convolutional_unit_weights(
                model.mixed_7a.branch0[1]
            ),
            branch_2_conv_1=extract_convolutional_unit_weights(
                model.mixed_7a.branch1[0]
            ),
            branch_2_conv_2=extract_convolutional_unit_weights(
                model.mixed_7a.branch1[1]
            ),
            branch_3_conv_1=extract_convolutional_unit_weights(
                model.mixed_7a.branch2[0]
            ),
            branch_3_conv_2=extract_convolutional_unit_weights(
                model.mixed_7a.branch2[1]
            ),
            branch_3_conv_3=extract_convolutional_unit_weights(
                model.mixed_7a.branch2[2]
            ),
        ),
        inception_resnet_c=(
            [
                InceptionResNetCWeights(
                    branch_1_conv=extract_convolutional_unit_weights(module.branch0),
                    branch_2_conv_1=extract_convolutional_unit_weights(
                        module.branch1[0]
                    ),
                    branch_2_conv_2=extract_convolutional_unit_weights(
                        module.branch1[1]
                    ),
                    branch_2_conv_3=extract_convolutional_unit_weights(
                        module.branch1[2]
                    ),
                    output_conv=extract_convolution_weights(module.conv2d),
                    # Not stored in the serialised state_dict (but defined in
                    # FaceNet-PyTorch)
                    residual_scale=0.20,
                )
                for module in model.repeat_3
            ]
            + [
                InceptionResNetCWeights(
                    branch_1_conv=extract_convolutional_unit_weights(
                        model.block8.branch0
                    ),
                    branch_2_conv_1=extract_convolutional_unit_weights(
                        model.block8.branch1[0]
                    ),
                    branch_2_conv_2=extract_convolutional_unit_weights(
                        model.block8.branch1[1]
                    ),
                    branch_2_conv_3=extract_convolutional_unit_weights(
                        model.block8.branch1[2]
                    ),
                    output_conv=extract_convolution_weights(model.block8.conv2d),
                    # Not stored in the serialised state_dict (but defined in
                    # FaceNet-PyTorch)
                    residual_scale=1.0,
                )
            ]
        ),
        output_dimension_reduction=LinearWeights(
            # NB: Weights stored transposed in PyTorch
            weights=n(model.last_linear.weight.T),
            biases=None,
        ),
        output_batch_normalisation=BatchNormalisationWeights(
            population_mean=n(model.last_bn.running_mean),
            population_variance=n(model.last_bn.running_var),
            weights=n(model.last_bn.weight),
            biases=n(model.last_bn.bias),
            # Hard-coded in FaceNet-PyTorch to match TensorFlow but not stored
            # in the serialised state_dict.
            eps=0.001,
        ),
    )


class StateDictAsObjects:
    """
    Present a PyTorch state dictionary (whose keys look like, e.g.
    repeat_3.4.branch1.2.bn.weight) as if it was an instantiated PyTorch module
    (e.g. make it accessible like this.repeat_3[4]branch1[2]bn.weight)
    """

    def __init__(self, state_dict: dict[str, Any], prefix: str = "") -> None:
        self._state_dict = state_dict
        self._prefix = prefix

    def __getattr__(self, key) -> Any:
        full_key = self._prefix + key

        # Try finding a leaf value with the correct name
        try:
            return self._state_dict[full_key]
        except KeyError:
            pass

        # Failing that, try to produce a nesting
        new_prefix = full_key + "."
        if any(k.startswith(new_prefix) for k in self._state_dict):
            return StateDictAsObjects(self._state_dict, new_prefix)

        raise AttributeError(key)

    def __getitem__(self, index) -> Any:
        try:
            return self.__getattr__(str(index))
        except AttributeError:
            raise IndexError(index)

    def __iter__(self) -> Any:
        for i in count():
            try:
                yield self[i]
            except IndexError:
                break


def main():
    parser = ArgumentParser(
        description="""
        Convert FaceNet weights from FaceNet-PyTorch format to that used by
        faceie.
        """
    )
    parser.add_argument("input", help="FaceNet-PyTorch weights file")
    parser.add_argument("output", help="Output (faceie-format weights) filename.")

    args = parser.parse_args()

    state_dict = torch.load(args.input)
    weights = extract_weights(StateDictAsObjects(state_dict))

    with open(args.output, "wb") as f:
        weightie.dump(weights, f)
