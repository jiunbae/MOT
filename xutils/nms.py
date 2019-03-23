import numpy as np
import torch


def nms(boxes: np.ndarray, scores: np.ndarray,
        overlap: float = .5, top: int = 0) \
        -> np.ndarray:
    """Apply non-maximum suppression

    Args:
        boxes: (tensor, (num, 4)) The location predictions for the image.
        scores: (tensor, (num)) The class prediction scores for the image.
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top: (int) The Maximum number of box predictions to consider. (0 is unlimited)
    Return:
        The indices of the kept boxes with respect to num.
    """
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)

    keep = scores.new(scores.size(0)).zero_().long()

    if boxes.numel() == 0:
        return keep

    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]

    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    if top:
        idx = idx[-top:]
    xx1, yy1, xx2, yy2 = [boxes.new() for _ in range(4)]
    w, h = boxes.new(), boxes.new()

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
        idx = idx[IoU.le(overlap)]

    return keep
