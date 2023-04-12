"""
Convert the MTCNN weights from FaceNet-PyTorch's format into the structure used
by this library.
"""

from argparse import ArgumentParser

import torch
import numpy as np

import pickle

from faceie.mtcnn.model import (
    PNetWeights,
    RNetWeights,
    ONetWeights,
    Conv2DWeights,
    LinearWeights,
)


def main():
    parser = ArgumentParser(
        description="""
        Convert MTCNN weights from FaceNet-PyTorch format to that used by
        faceie.
        """
    )
    parser.add_argument(
        "--p-net",
        "-p",
        nargs=2,
        metavar=("input", "output"),
    )
    parser.add_argument(
        "--r-net",
        "-r",
        nargs=2,
        metavar=("input", "output"),
    )
    parser.add_argument(
        "--o-net",
        "-o",
        nargs=2,
        metavar=("input", "output"),
    )

    args = parser.parse_args()

    ################################################################################

    if args.p_net is not None:
        state_dict = torch.load(args.p_net[0])

        # The bounding box coordinates are given relative to the input top-left and
        # bottom-right corners respectively in the PyTorch implementation. I'd prefer
        # them both to be relative to the top-left corner coordinates
        bounding_boxes_biases = state_dict["conv4_2.bias"].numpy()  # (4)
        bounding_boxes_biases[2:] += 1

        weights = PNetWeights(
            conv1=Conv2DWeights(
                weights=state_dict["conv1.weight"].numpy(),
                biases=state_dict["conv1.bias"].numpy(),
            ),
            prelu1=state_dict["prelu1.weight"].numpy(),
            conv2=Conv2DWeights(
                weights=state_dict["conv2.weight"].numpy(),
                biases=state_dict["conv2.bias"].numpy(),
            ),
            prelu2=state_dict["prelu2.weight"].numpy(),
            conv3=Conv2DWeights(
                weights=state_dict["conv3.weight"].numpy(),
                biases=state_dict["conv3.bias"].numpy(),
            ),
            prelu3=state_dict["prelu3.weight"].numpy(),
            classifier=Conv2DWeights(
                weights=state_dict["conv4_1.weight"].numpy(),
                biases=state_dict["conv4_1.bias"].numpy(),
            ),
            bounding_boxes=Conv2DWeights(
                weights=state_dict["conv4_2.weight"].numpy(),
                biases=bounding_boxes_biases,
            ),
        )

        pickle.dump(weights, open(args.p_net[1], "wb"))

    ################################################################################

    if args.r_net is not None:
        state_dict = torch.load(args.r_net[0])

        # NB: The final dense layer in the PyTorch model assumes pixels are ordered
        # (width, height, channels) however, in our implementation by convention we use
        # (channels, height, width). As such we shuffle the matrix around now rather
        # than shuffling pixel data every time we use it...
        #
        # NB: Numpy linear transform weights stored transposed by PyTorch
        dense4_weight = state_dict["dense4.weight"].numpy()  # (128, 3*3*64)
        dense4_weight = dense4_weight.reshape(128, 3, 3, 64)  # (128, 3, 3, 64)
        dense4_weight = np.moveaxis(
            dense4_weight, (1, 2, 3), (3, 2, 1)
        )  # (128, 64, 3, 3)
        dense4_weight = dense4_weight.reshape(128, 64 * 3 * 3)  # (128, 64*3*3)
        dense4_weight = dense4_weight.T  # (64*3*3, 128)
        dense4_weight = np.ascontiguousarray(dense4_weight)

        # The bounding box coordinates are given relative to the input top-left and
        # bottom-right corners respectively in the PyTorch implementation. I'd prefer
        # them both to be relative to the top-left corner coordinates
        bounding_boxes_biases = state_dict["dense5_2.bias"].numpy()  # (4)
        bounding_boxes_biases[2:] += 1

        weights = RNetWeights(
            conv1=Conv2DWeights(
                weights=state_dict["conv1.weight"].numpy(),
                biases=state_dict["conv1.bias"].numpy(),
            ),
            prelu1=state_dict["prelu1.weight"].numpy(),
            conv2=Conv2DWeights(
                weights=state_dict["conv2.weight"].numpy(),
                biases=state_dict["conv2.bias"].numpy(),
            ),
            prelu2=state_dict["prelu2.weight"].numpy(),
            conv3=Conv2DWeights(
                weights=state_dict["conv3.weight"].numpy(),
                biases=state_dict["conv3.bias"].numpy(),
            ),
            prelu3=state_dict["prelu3.weight"].numpy(),
            linear=LinearWeights(
                # NB: Linear transform weights stored transposed by PyTorch
                dense4_weight,
                biases=state_dict["dense4.bias"].numpy(),
            ),
            prelu4=state_dict["prelu4.weight"].numpy(),
            classifier=LinearWeights(
                # NB: Linear transform weights stored transposed by PyTorch
                weights=state_dict["dense5_1.weight"].T.numpy(),
                biases=state_dict["dense5_1.bias"].numpy(),
            ),
            bounding_boxes=LinearWeights(
                # NB: Linear transform weights stored transposed by PyTorch
                weights=state_dict["dense5_2.weight"].T.numpy(),
                biases=bounding_boxes_biases,
            ),
        )

        pickle.dump(weights, open(args.r_net[1], "wb"))

    ################################################################################

    if args.o_net is not None:
        state_dict = torch.load(args.o_net[0])

        # NB: The final dense layer in the PyTorch model assumes pixels are ordered
        # (width, height, channels) however, in our implementation by convention we use
        # (channels, height, width). As such we shuffle the matrix around now rather
        # than shuffling pixel data every time we use it...
        #
        # NB: Numpy linear transform weights stored transposed by PyTorch
        dense5_weight = state_dict["dense5.weight"].numpy()  # (256, 3*3*128)
        dense5_weight = dense5_weight.reshape(256, 3, 3, 128)  # (256, 3, 3, 128)
        dense5_weight = np.moveaxis(
            dense5_weight, (1, 2, 3), (3, 2, 1)
        )  # (256, 128, 3, 3)
        dense5_weight = dense5_weight.reshape(256, 128 * 3 * 3)  # (256, 128*3*3)
        dense5_weight = dense5_weight.T  # (128*3*3, 256)
        dense5_weight = np.ascontiguousarray(dense5_weight)

        # The bounding box coordinates are given relative to the input top-left and
        # bottom-right corners respectively in the PyTorch implementation. I'd prefer
        # them both to be relative to the top-left corner coordinates
        bounding_boxes_biases = state_dict["dense6_2.bias"].numpy()  # (4)
        bounding_boxes_biases[2:] += 1

        # NB: The dense6_3 transform orders its outputs as x1, x2, x3, x4, x5, y1, y2,
        # y3, y4, y5 which is inconsistent with the bounding box regressions which are
        # ordered x1, y1, x2, y2, etc. As a result we re-order the matrix here to
        # produce a better output ordering.
        #
        # NB: Numpy linear transform weights stored transposed by PyTorch
        dense6_3_weights = state_dict["dense6_3.weight"].T.numpy()  # (256, 10)
        dense6_3_weights = dense6_3_weights.reshape(256, 2, 5)  # (256, 2, 5)
        dense6_3_weights = np.moveaxis(dense6_3_weights, (1, 2), (2, 1))  # (256, 5, 2)
        dense6_3_weights = dense6_3_weights.reshape(256, 10)  # (256, 10) (reordered)

        dense6_3_biases = state_dict["dense6_3.bias"].numpy()  # (10)
        dense6_3_biases = dense6_3_biases.reshape(2, 5)  # (2, 5)
        dense6_3_biases = np.moveaxis(dense6_3_biases, 0, 1)  # (5, 2)
        dense6_3_biases = dense6_3_biases.reshape(10)  # (10) (reordered)

        weights = ONetWeights(
            conv1=Conv2DWeights(
                weights=state_dict["conv1.weight"].numpy(),
                biases=state_dict["conv1.bias"].numpy(),
            ),
            prelu1=state_dict["prelu1.weight"].numpy(),
            conv2=Conv2DWeights(
                weights=state_dict["conv2.weight"].numpy(),
                biases=state_dict["conv2.bias"].numpy(),
            ),
            prelu2=state_dict["prelu2.weight"].numpy(),
            conv3=Conv2DWeights(
                weights=state_dict["conv3.weight"].numpy(),
                biases=state_dict["conv3.bias"].numpy(),
            ),
            prelu3=state_dict["prelu3.weight"].numpy(),
            conv4=Conv2DWeights(
                weights=state_dict["conv4.weight"].numpy(),
                biases=state_dict["conv4.bias"].numpy(),
            ),
            prelu4=state_dict["prelu4.weight"].numpy(),
            linear=LinearWeights(
                # NB: Linear transform weights stored transposed by PyTorch
                dense5_weight,
                biases=state_dict["dense5.bias"].numpy(),
            ),
            prelu5=state_dict["prelu5.weight"].numpy(),
            classifier=LinearWeights(
                # NB: Linear transform weights stored transposed by PyTorch
                weights=state_dict["dense6_1.weight"].T.numpy(),
                biases=state_dict["dense6_1.bias"].numpy(),
            ),
            bounding_boxes=LinearWeights(
                # NB: Linear transform weights stored transposed by PyTorch
                weights=state_dict["dense6_2.weight"].T.numpy(),
                biases=bounding_boxes_biases,
            ),
            landmarks=LinearWeights(
                # NB: Linear transform weights stored transposed by PyTorch
                weights=dense6_3_weights,
                biases=dense6_3_biases,
            ),
        )

        pickle.dump(weights, open(args.o_net[1], "wb"))
