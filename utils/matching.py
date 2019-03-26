from typing import Tuple, List

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.linear_assignment_ import linear_assignment

from lib.trace import Trace
from utils import box


def iou_distance(first: List[Trace], second: List[Trace]) \
        -> np.ndarray:
    """Compute cost based on IoU

    Args:
        first:
        second:

    Returns: cost_matrix
    """
    unions = np.zeros((len(first), len(second)), dtype=np.float)

    if unions.size:
        unions = box.iou(
            np.ascontiguousarray([track.to_tlbr for track in first], dtype=np.float),
            np.ascontiguousarray([track.to_tlbr for track in second], dtype=np.float)
        )

    return 1 - unions


def nearest_distance(tracks: list, detections: list, metric='cosine') \
        -> np.ndarray:
    """Compute cost based on ReID features

    Args:
        tracks:
        detections:
        metric:

    Returns: cost matrix
    """
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    if cost.size:
        features = np.asarray(list(map(lambda t: t.feature_current, detections)), dtype=np.float64)
        for index, track in enumerate(tracks):
            cost[index, :] = np.maximum(0.0, cdist(track.features, features, metric).min(axis=0))

    return cost


def gate_cost(motion, cost: np.ndarray,
              tracks: list, detections: list, only_position: bool = False) \
        -> np.ndarray:
    """Gate cost matrix

    Args:
        motion:
        cost:
        tracks:
        detections:
        only_position:

    Returns: cost matrix
    """
    if cost.size:
        dimension = 2 if only_position else 4
        threshold = motion.threshold[dimension]
        measurements = np.stack(list(map(lambda d: box.calibrate(d.to_tlwh), detections)))

        for index, track in enumerate(tracks):
            distance = motion.gating_distance(
                measurements,
                track.mean,
                track.cov,
                only_position
            )
            cost[index, distance > threshold] = np.inf

    return cost


def assignment(cost: np.ndarray, threshold: float, epsilon: float = 1e-4) \
        -> Tuple[np.ndarray, Tuple[int], Tuple[int]]:
    if not cost.size:
        return np.empty((0, 2), dtype=int), \
               tuple(range(np.size(cost, 0))), \
               tuple(range(np.size(cost, 1)))

    cost[cost > threshold] = threshold + epsilon
    indices = linear_assignment(cost)
    matches = indices[cost[tuple(zip(*indices))] <= threshold]

    return matches, \
           tuple(set(range(np.size(cost, 0))) - set(matches[:, 0])), \
           tuple(set(range(np.size(cost, 1))) - set(matches[:, 1]))
