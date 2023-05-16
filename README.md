Faceie: A Pure-Numpy Face Recognition Library
=============================================

Faceie is an inference-only, CPU-only, pure [Numpy](https://numpy.org/)
implementation of MTCNN face detection and an Inception-ResNet-v1 based,
FaceNet-style face encoder. Both components are intended to be
weight-compatible with
[FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch).


Some more detailed notes (and references) on the two models can be found here:

* [MTCNN (Face Detection)](./faceie/mtcnn/__init__.py)
* [Inception-ResNet-v1/FaceNet (Face Embedding)](./faceie/facenet/__init__.py)


Usage
-----

    >>> from faceie import detect_faces, encode_faces
    
    >>> from PIL import Image
    >>> image = image.load("path/to/image.jpg")
    
    >>> # Use MTCNN to find faces in an image (the returned named tuple
    >>> # includes probabilities, bounding boxes, facial landmark coordinates
    >>> # and thumbnails of each face by default).
    >>> faces = detect_faces(image)
    
    >>> # Produce a 512-dimensional embedding for each detected face
    >>> embeddings = encode_faces(faces.images)
    
    >>> # Compare detected faces against a target face using squared-distance
    >>> # metric. Smaller distances (around 1 or less) imply the same face.
    >>> import numpy as np
    >>> target = image.load("path/to/target.jpg")
    >>> target_embedding = encode_faces(detect_faces(target).images[0])
    >>> square_dists = np.sum(np.power(embeddings - target_embedding, 2), axis=1)


Weights
-------

This implementation is weight-compatible with
[FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) but uses its own
file formats (primarily to avoid the PyTorch dependency).

The MTCNN model weights are very small (a couple of megabytes) and are stored
[included with Faceie's source](./faceie/mtcnn/data/).

The Inception-ResNet-v2 based FaceNet-style network model weights are larger
(around one hundred megabytes) and are distributed separately. By default,
Faceie will automatically download and use a copy of the VGGFace-trained
weights (preconverted into Faceie's format) from the corresponding [Faceie
GitHub release](https://github.com/mossblaser/faceie/releases/).  (See
[weightie](https://github.com/mossblaser/weightie) for the weight storage
format and auto-download mechanism.).

You can also use the `faceie-convert-facenet-weights` script to convert a
PyTorch weights file from the reference
[FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) implementation's
format into the native weights format:

    $ pip install path/to/faceie[convert]  # Extra packages needed for weights file conversion
    $ faceie-convert-facenet-weights /path/to/weights.pt output.weights

The conversion script requires extra packages to be installed in order to
unpack the PyTorch weights file format (including PyTorch). After conversion,
these dependencies are no longer required.

If using a custom weights file, you can provide the path to the converted
weights file (as a [Path](https://docs.python.org/3/library/pathlib.html)
object) to `faceie.facenet.load` to load the needed weights and pass them to the
encoder functions:

    >>> # Load the weights...
    >>> from pathlib import Path
    >>> from faceie.facenet import load
    >>> weights = load(Path("/path/to/converted.weights"))
    
    >>> # Use them...
    >>> embeddings = encode_faces(..., weights)


Preemptive FAQ
--------------

**Why does Faceie exist?**

As with [Clippie](https://github.com/mossblaser/clippie/), I am gradually
building my own search facilities for my own photo collection which doesn't
depend on 3rd party services (e.g. Google Photos). The MTCNN and
Inception-ResNet-v1 FaceNet networks hit a sweet spot:

* Both are fairly well documented in the literature
* Both produce fairly robust detections/embeddings respectively. Compared with
  much faster popular models and algorithms, this combination does a good job
  at handling differing facial poses, orientations, lighting and so on.
  Compared with the current state of the art, they're not *that* far behind. In
  any case, in my casual experimentation, they seem to be generally as good as
  or better than the functions built into popular photo management services and
  software.
* Both run "well enough" on a CPU to be practical for a personal photo
  collection.
* Both have readily available pre-trained models available under permissive
  licenses. Specifically
  [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch) includes both!

As for the obvious follow-up question...

**Why not use [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch)?**

By contrast with FaceNet-PyTorch, Faceie has only a few comparatively
light-weight and stable dependencies (chiefly [Numpy](https://numpy.org/) and
[Pillow](https://pillow.readthedocs.io/en/stable/)). As such, the largest
download needed is a copy of the weights, not gigabytes of software.
Furthermore, unlike most deep learning libraries -- which cater to a fast
moving field -- all of the dependencies used have been stable and well
supported for many years and are likely to remain so for many more years.

With that said,
[FaceNet-PyTorch]((https://github.com/timesler/facenet-pytorch)) is a really
cool project with a track record of active development and ongoing
improvements. It is around twice as fast as Faceie run on my CPU and
dramatically faster running on a GPU -- which Faceie doesn't support. If you're
not me, you should probably use FaceNet-PyTorch!

Last but not least, re-implementing things like this is a good way for me to
get to know how they work since I'm entirely new to the field!


**Why CPU only?**

Faceie can detect a set of faces in a photograph and produce 512 embeddings for
them in a second or two on my laptop. This is adequately fast for my use. Even
at this (quite pitiful) speed, processing my lifetimes' back-catalogue is just
a one-off few days of CPU time. In any case, if speed was a real concern, I'd
definitely be looking at other solutions.

Faceie's Numpy based implementation runs approximately twice as slowly as the
PyTorch-based implementation on a CPU. This is obviously a bit disappointing (I
see no reason why this should be the case) but I'm insufficiently motivated to
dig deeper.


**Why inference only?**

Since I'm only interested in *using* these models, and published weights work
well, I had no need. I'm also especially keen to avoid falling down the rabbit
hole of model training...


**Does Faceie include any clustering algorithms?**

No, this is out of scope for now.


**Does this reuse any code from FaceNet-PyTorch?**

No -- though it does use its model weights.

This software is a from-scratch implementation of MTCNN and
Inception-ResNet-v1/FaceNet primarily based on their respective papers.
However, for the sake of weight compatibility, some parts necessarily mimic the
behaviour of FaceNet-PyTorch.

Whilst no code is reused, Faceie directly uses the pretrained weights
distributed by FaceNet-PyTorch under the [MIT
License](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md):

* The (converted) MTCNN model weights are included in this repository in
  [`faceie/mtcnn/data`](faceie/mtcnn/data)

* The (converted) Inception-ResNet-v1/FaceNet weights are included alongside
  [GitHub releases](https://github.com/mossblaser/faceie/releases/) of Faceie.

