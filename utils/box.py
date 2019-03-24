import numpy as np


def calibrate(box: np.ndarray) \
        -> np.ndarray:
    result = box.copy()
    result[:2] += result[2:] / 2
    result[2] /= result[3]
    return result


def calibration(func):
    def wrapper(box, *args, **kwargs):
        box = calibrate(box)
        return func(box, *args, **kwargs)
    return wrapper


def iou(first: np.ndarray, second: np.ndarray) \
        -> np.ndarray:

    intersection = np.zeros((first.shape[0], second.shape[0]))

    for k in range(second.shape[0]):
        second_area = (second[k, 2] - second[k, 0] + 1) * (second[k, 3] - second[k, 1] + 1)
        for n in range(first.shape[0]):
            iw = min(first[n, 2], second[k, 2]) - max(first[n, 0], second[k, 0]) + 1
            if iw > 0:
                ih = min(first[n, 3], second[k, 3]) - max(first[n, 1], second[k, 1]) + 1
                if ih > 0:
                    first_area = (first[n, 2] - first[n, 0] + 1) * (first[n, 3] - first[n, 1] + 1)
                    inter_area = iw * ih
                    intersection[n, k] = inter_area/(second_area+first_area-inter_area)

    return intersection


def clip(boxes: np.ndarray, shape: tuple) \
        -> np.ndarray:
    """Clip boxes to image boundaries

    Args:
        boxes:
        shape:

    Returns: clipped boxes
    """

    if not np.size(boxes, 0):
        return boxes

    boxes = np.copy(boxes)

    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], shape[0] - 1), 0)

    return boxes
