"""
An implementation of the Non-Maximum Suppression algorithm for removing
(approximate) duplicates from a set of overlapping rectangles.

.. autofunction:: non_maximum_suppression
"""

import numpy as np
from numpy.typing import NDArray


def non_maximum_suppression(
    probabilities: NDArray,
    bounding_boxes: NDArray,
    maximum_iou: float,
) -> NDArray:
    """
    Performs non-maximum suppression (NMS) on a collection of bounding boxes
    and associated probabilities. This returns the subset of bounding boxes
    which are both sufficiently non-overlapping.

    Parameters
    ==========
    probabilities : array (num_candidates)
        An array of probabilities of each candidate bounding box containing a
        face.
    bounding_boxes : array (num_candidates, 4)
        The corresponding bounding boxes which may contain a face, for the same
        shape of values as ``probabilities``. The last axis contains the
        bounding box corners as x1, y1, x2 and y2 respectively.
    maximum_iou : float
        The maximum overlapping area (as Intersection over Union (IoU)) two
        bounding boxes can have before the lower-probability box is removed.

    Returns
    =======
    array (num_results)
        An array of indices into the input ``probabilities`` and
        ``bounding_boxes`` arrays which indicate the bounding boxes selected by
        the non-maximum suppression algoirthm.
    """
    # We proceed by repeatedly selecting the most likely bounding box and then
    # removing any others which overlap too much with it.
    selected_indices = []
    candidate_indices = np.argsort(probabilities)
    while len(candidate_indices) > 0:
        # Find (and keep) most likley face
        best_index = candidate_indices[-1]
        selected_indices.append(best_index)
        candidate_indices = candidate_indices[:-1]

        # Now lets find (and remove) any faces which overlap with this face by
        # too much.

        # Bounding box of most likely ('Best') face
        bx1, by1, bx2, by2 = bounding_boxes[best_index]

        # Bounding boxes of all 'Other' candidate faces
        ox1s, oy1s, ox2s, oy2s = bounding_boxes[candidate_indices].T

        # Work out intersecting areas
        #
        # In 1D, consider two pairs of coordinates with an intersecting region:
        #
        #     ax1|----------|ax2
        #         bx1|--------------|bx2
        #         ix1|------|ix2
        #
        # The intersecting region (denoted by ix1 and ix2 above) is given by
        #
        #     ix1 = max(ax1, bx1)
        #     ix2 = min(ax2, bx2)
        #
        # There is an intersection whenever ix1 < ix2, otherwise the regions do
        # not intersect.
        #
        # So, lets carry this out en-masse on our bounding box vs all other
        # bounding boxes -- and for both x and y axes:
        intersection_x1s = np.maximum(bx1, ox1s)
        intersection_x2s = np.minimum(bx2, ox2s)
        intersection_xs = np.maximum(0, intersection_x2s - intersection_x1s)

        intersection_y1s = np.maximum(by1, oy1s)
        intersection_y2s = np.minimum(by2, oy2s)
        intersection_ys = np.maximum(0, intersection_y2s - intersection_y1s)

        intersection_areas = intersection_xs * intersection_ys

        # Now lets compute the total area of the union of the best bounding box
        # and all other bounding boxes.
        best_area = (bx2 - bx1) * (by2 - by1)
        other_areas = (ox2s - ox1s) * (oy2s - oy1s)
        union_areas = best_area + other_areas - intersection_areas

        # Finally, lets compute the intersection-over-union (IoU) and discard
        # any candidates with excessive overlap
        ious = intersection_areas / union_areas
        candidate_indices = candidate_indices[np.where(ious <= maximum_iou)]

    return np.array(selected_indices, dtype=int)
