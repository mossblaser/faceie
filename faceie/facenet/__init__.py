"""
A (Numpy-based) implementation of face-to-vector encoding based on the
FaceNet methodology but using an Inception-ResNet-v1 network architecture
based on "FaceNet-PyTorch" (https://github.com/timesler/facenet-pytorch).

Background
==========

The FaceNet methodology for producing a face-to-vector system was introduced in
"FaceNet: A Unified Embedding for Face Recognition and Clustering" by Schroff,
Kalenichenko and Philbin (2015). It describes a collection of networks,
including several based on the 'Inception-v1' convolutional network
architecture capable of producing state-of-the-art face vector embeddings
suitable for various tasks.

The Inception-ResNet-v1 architecture was introduced in "Inception-v4,
Inception-ResNet and the Impact of Residual Connections on Learning" by
Szegedy, Ioffe, Vanhoucke and Alemi (2016). These Inception architectures form
the forth generation of refinements boasting improvements in performance and
training speed.

This implementation is intended to be weight-compatible with Tim Esler's
FaceNet-PyTorch implementation (https://github.com/timesler/facenet-pytorch).
This, in turn, was built on weights ported from David Sandberg's FaceNet
implementation based on Tensorflow (https://github.com/davidsandberg/facenet).
The latter began life as an implementation of the FaceNet paper's Inception-v1
based "NN3" model, hence the 'FaceNet' name. Later on, this project switched
over to using the Inception-ResNet-v1 architecture. It is this network which
FaceNet-PyTorch eventually re-implemented and so to what this module
reimplements using Numpy. Inception indeed.

"""

from faceie.facenet.model import encode_faces, FaceNetWeights, load
