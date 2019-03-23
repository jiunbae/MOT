import numpy as np


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
