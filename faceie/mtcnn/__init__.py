"""
A (Numpy-based) implementation of face detection based on Multi-Task Cascaded
Convolutional Networks (MTCNN).

.. autofunction:: detect_faces

.. autoclass:: DetectedFaces


Background
==========

This technique was originally published by Zhang et al. in "Joint Face
Detection and Alignment using Multi-task Cascaded Convolutional Networks"
(2016). This implementation attempts to be weight-compatible with Facenet
PyTorch (https://github.com/timesler/facenet-pytorch) since we use their
weights. In all other respects, this implementation is independent of others
and does not reuse any code.

Whilst the (internal) convolutional networks do produce identical results to
Facenet-PyTorch's networks, the other parts of the implementation differ
slightly either due to different default hyperparameters or just small
differences in implementation.

As an aside, I found that other implementations were frequently unclear and
difficult to follow (in my humble opinion, as an outsider). As a result, I've
tried to make this implementation somewhat easier to follow -- or at least more
explicit.


Performance
===========

Performance-wise this (CPU-only) implementation is somewhat slower
than the (CPU-based) one in Facenet-PyTorch by a factor of around 2-3. This
largely due to certain Numpy operations (in particular ``np.amax`` used in the
max pooling layers) not automatically parallelising across CPU cores. Inserting
some fairly crude parallelism brought the speed difference down to 1-2 times
slower but since I'd rather keep things clean (it is already "fast enough" for
my uses) I've not pursued this line.
"""

from faceie.mtcnn.detect_faces import detect_faces, DetectedFaces
