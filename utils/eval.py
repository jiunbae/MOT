from typing import Tuple

import numpy as np


def evaluate_frame(ground_truths: np.ndarray, detections: np.ndarray,
                   threshold: float = .2) \
        -> Tuple[float, float]:
    pass


def evaluate(ground_truths: np.ndarray, detections: np.ndarray,
             threshold: float = .2) \
        -> Tuple[float, float]:
    """ Evaluation tracking
    Args:
        ground_truths (ndarray)[?, (frame, id, x, y, w, h)]:
        detections (ndarray)[?, (frame, id, x, y, w, h)]:
    Returns:
        - MOTA Score (float)
        - MOTP Score (float)

    """

    frames = ground_truths[:, 0]
    axis = np.concatenate([
        np.where(np.diff(frames) > 0)[0],
        [np.size(frames, 0) - 1],
    ])

    pass
