from collections import deque
from enum import Enum

import numpy as np


class State(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Replaced = 4


class Trace(object):
    count = 0

    def __init__(self, box: np.ndarray, score: float, max_feature: int = 100, from_det: bool = True):
        self.state = State.New
        self.id = 0
        self.begin = self.frame = 0

        self.box = box
        self.score = score
        self.max_feature = max_feature

        self.features = deque([], maxlen=self.max_feature)
        self.feature_current = self.feature_last = None
        self.mean = self.cov = None

        self.update_time = 0
        self.is_activated = False

        # classification
        self.from_det = from_det
        self.tracked_length = 0
        self.tracked_time = 0

        # Motion Model
        self.motion = None

        # TODO: append single object tracker
        self.tracker = None

    def tracking(self, image: np.ndarray) \
            -> np.ndarray:
        return self.tracker.predict(image) if self.tracker else self.box

    def predict(self):
        if self.update_time > 0:
            self.tracked_length = 0

        self.update_time += 1

        mean = self.mean.copy()
        if self.state != State.Tracked:
            mean[7] = 0
        self.mean, self.cov = self.motion.predict(mean, self.cov)

        if self.tracker:
            self.tracker.update_roi(self, self.box)

    def activate(self, frame: int, image: np.ndarray, motion):
        self.state = State.Tracked
        self.id = self.next()
        self.begin = self.frame = frame

        self.motion = motion
        self.mean, self.cov = self.motion.initialize(self.box)

    def reactivate(self, frame: int, image: np.ndarray, trace, reassign: bool = False):
        self.state = State.Tracked
        self.frame = frame

        self.update_time = 0
        self.tracked_time = 0
        self.tracked_length = 0
        self.is_activated = True

        self.mean, self.cov = self.motion.update(
            self.mean, self.cov, trace.box
        )

        if reassign:
            self.id = self.next()

        self.feature = trace.feature_current

    def update(self, frame: int, image: np.ndarray, trace, update_feature: bool = True):
        self.state = State.Tracked
        self.frame = frame

        self.update_time = 0
        if trace.from_det:
            self.tracked_time = 0
        else:
            self.tracked_time += 1
        self.tracked_length += 1

        self.mean, self.cov = self.motion.update(
            self.mean, self.conv, trace.box
        )
        self.is_activated = True

        self.score = trace.score

        if update_feature:
            self.feature = trace.feature_current
            if self.tracker:
                self.tracker.update(image, self.box)

    @staticmethod
    def next():
        Trace.count += 1
        return Trace.count

    @property
    def track_score(self):
        return max(0, 1 - np.log(1 + .05 * self.tracked_time)) * (self.tracked_length - self.tracked_time > 2)

    @property
    def feature(self):
        return self.features

    @feature.setter
    def feature(self, f):
        if f is not None:
            self.feature_current = self.feature_last = f
            self.features.append(f)

    @property
    def to_tlwh(self):
        if self.mean is None:
            return self.box.copy()

        result = self.mean[:4].copy()
        result[2] *= result[3]
        result[:2] -= result[2:] / 2
        return result

    @property
    def to_tlbr(self):
        result = self.to_tlwh.copy()
        result[2:] += result[:2]

        return result
