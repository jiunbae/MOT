from typing import Iterator

import numpy as np
import torch


# TODO: Remove this parts
def nms(*args, **kwargs):
    return torch_nms(*args, **kwargs)


def cpu_nms(detections: np.ndarray, scores: np.ndarray,
            threshold: float = .5) \
        -> Iterator[int]:
    """Apply non-maximum suppression

    Arguments:
        detections: (tensor, (num, 4)) The location predictions for the image.
        scores: (tensor, (num)) The class prediction scores for the image.
        threshold: (float) The overlap thresh for suppressing unnecessary boxes.
    Return:
        The indices of the kept boxes with respect to num.
    """
    x1, x2 = detections[:, 0], detections[:, 2]
    y1, y2 = detections[:, 1], detections[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]

        yield i

        xx1, xx2 = np.maximum(x1[i], x1[order[1:]]), np.minimum(x2[i], x2[order[1:]])
        yy1, yy2 = np.maximum(y1[i], y1[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(overlap <= threshold)[0] + 1]


def torch_nms(detections: np.ndarray, scores: np.ndarray,
              threshold: float = .5, top: int = 0) \
        -> np.ndarray:
    """Apply non-maximum suppression based on Torch

    Arguments:
        detections: (tensor, (num, 4)) The location predictions for the image.
        scores: (tensor, (num)) The class prediction scores for the image.
        threshold: (float) The overlap thresh for suppressing unnecessary boxes.
        top: (int) slice top k
    Return:
        The indices of the kept boxes with respect to num.
    """
    detections = torch.from_numpy(detections)
    scores = torch.from_numpy(scores)

    keep = scores.new(scores.size(0)).zero_().long()

    if detections.numel() == 0:
        return keep

    x1, y1 = detections[:, 0], detections[:, 1]
    x2, y2 = detections[:, 2], detections[:, 3]

    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    if top:
        idx = idx[-top:]
    xx1, yy1, xx2, yy2 = [detections.new() for _ in range(4)]
    w, h = detections.new(), detections.new()

    count = 0
    while idx.numel() > 0:
        keep[count] = i = idx[-1]
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w, h = xx2 - xx1, yy2 - yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h

        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union
        idx = idx[IoU.le(threshold)]

    return keep
