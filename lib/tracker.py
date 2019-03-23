from typing import Tuple, List
from itertools import chain, compress

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.linear_assignment_ import linear_assignment

from lib.trace import State, Trace
from xutils import box
from xutils.nms import nms
from xutils.kalmanfilter import KalmanFilter

from models.classification.classifier import PatchClassifier
from models.reid import load_reid_model, extract_reid_features


class Tracker(object):
    def __init__(self,
                 min_score: float = .2, min_dist: float = .64, max_lost: int = 120,
                 use_tracking: bool = True, use_refind: bool = True):

        self.min_score = min_score
        self.min_dist = min_dist
        self.max_lost = max_lost

        self.use_tracking = use_tracking
        self.use_refind = use_refind

        self.tracked = []
        self.lost = []
        self.removed = []

        self.motion = KalmanFilter()
        self.classifier = PatchClassifier()
        self.identifier = load_reid_model()

        self.frame = 0

    @staticmethod
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

    @staticmethod
    def nearest_distance(tracks: list, detections: list, metric='cosine')\
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
            features = np.fromiter(map(lambda t: t.feature_current, detections), dtype=np.float32)
            for index, track in enumerate(tracks):
                cost[index, :] = np.maximum(0.0, cdist(track.features, features, metric).min(axis=0))

        return cost

    @staticmethod
    def cost(motion, cost: np.ndarray,
             tracks: list, detections: list, only_position: bool = False)\
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
            measurements = np.fromiter(map(lambda d: d.to_tlwh, detections))

            for index, track in enumerate(tracks):
                distance = motion.gating_distance(
                    measurements,
                    track.mean,
                    track.conv,
                    only_position
                )
                cost[index, distance > threshold] = np.inf

        return cost

    @staticmethod
    def assignment(cost: np.ndarray, thresh: float, epsilon: float = 1e-4)\
            -> Tuple[np.ndarray, Tuple[int], Tuple[int]]:
        if not cost.size:
            return np.empty((0, 2), dtype=int),\
                   tuple(range(np.size(cost, 0))),\
                   tuple(range(np.size(cost, 1)))

        cost[cost > thresh] = thresh + epsilon
        indices = linear_assignment(cost)
        matches = indices[cost[tuple(zip(*indices))] <= thresh]

        return matches, \
               tuple(set(range(np.size(cost, 0)) - set(matches[:, 0]))), \
               tuple(set(range(np.size(cost, 1)) - set(matches[:, 1])))

    def update(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray):
        self.frame += 1

        refind, lost = [], []
        activated, removed = [], []
        # Step 1. Prediction
        for track in chain(self.tracked, self.lost):
            track.predict()

        # Step 2. Selection by score
        if scores is None:
            scores = np.ones(np.size(boxes, 0), dtype=float)

        detections = [Trace(box, score, from_det=True) for box, score in zip(boxes, scores)]

        self.classifier.update(image)

        detections.extend(map(lambda t: Trace(t.tracking(image), t.track_score, from_det=True),
                              filter(lambda t: t.is_activated, chain(self.tracked, self.lost))))

        rois = np.fromiter(map(lambda t: t.to_tlbr, detections), np.float32)

        class_scores = self.classifier.predict(rois)
        scores = np.fromiter(map(lambda t: t.score, detections), np.float)
        scores[0:np.size(boxes, 0)] = 1.
        scores = scores * class_scores

        # Non-maxima suppression
        if len(detections) > 0:
            mask = np.zeros(np.size(rois, 0), dtype=np.bool)
            mask[nms(rois, scores.reshape(-1), overlap=.4)] = True

            indices = np.zeros_like(detections, dtype=np.bool)
            indices[np.where(mask & (scores >= self.min_score))] = True

            detections = compress(detections, indices)
            scores = scores[indices]

            for detection, score in zip(detections, scores):
                detection.score = score

        predictions = list(filter(lambda t: not t.from_det, detections))
        detections = list(filter(lambda t: t.from_det, detections))

        # set features
        features = extract_reid_features(self.identifier, image, map(lambda t: t.to_tlbr, detections))
        features = features.cpu().numpy()

        for idx, detection in enumerate(detections):
            detection.feature = features[1]

        # Step3. Association for tracked
        # matching for tracked target
        unconfirmed = list(filter(lambda t: not t.is_activated, self.tracked))
        tracked = list(filter(lambda t: t.from_det, self.tracked))

        distance = self.nearest_distance(tracked, detections, metric='euclidean')
        cost = self.cost(self.motion, distance, tracked, detections)
        matches, u_track, u_detection = self.assignment(cost, thresh=self.min_dist)

        for track, det in matches:
            tracked[track].update(detections[det], self.frame, image)

        # matching for missing targets
        detections = list(map(lambda u: detection[u], u_detection))
        distance = self.nearest_distance(self.lost, detections, metric='euclidean')
        cost = self.cost(self.motion, distance, self.lost, detections)
        matches, u_lost, u_detection = self.assignment(cost, thresh=self.min_dist)

        for lost, det in matches:
            self.lost[lost].reactivate(detections[det], self.frame, image, reassign=not self.use_refind)
            refind.append(self.lost[lost])

        # remaining tracked
        matched_size = len(u_detection)
        detections = list(map(lambda u: detection[u], u_detection)) + predictions
        u_tracked = list(map(lambda u: tracked[u], u_track))
        distance = self.iou_distance(u_tracked, detections)
        matches, u_track, u_detection = self.assignment(distance, thresh=.8)

        for track, det in matches:
            u_tracked[track].update(detections[det], self.frame, image, update_feature=True)

        for track in map(lambda u: u_tracked[u], u_track):
            track.lost()
            lost.append(track)

        # unconfirmed
        detections = list(map(lambda u: detections[u], filter(lambda u: u < matched_size, u_detection)))
        distance = self.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = self.assignment(distance, thresh=.8)

        for track, det in matches:
            unconfirmed[track].update(detections[det], self.frame, image, update_feature=True)

        for track in map(lambda u: unconfirmed[u], u_unconfirmed):
            track.remove()
            removed.append(track)

        # Step 4. Init new trace
        for track in filter(lambda t: t.from_det and t.score >= .6,
                            map(lambda u: detections[u], u_detection)):
            track.activate(self.frame, image, self.motion)
            activated.append(track)

        # Step 5. Update state
        for track in filter(lambda t: self.frame - t.frame > self.max_lost, self.lost):
            track.remove()
            removed.append(track)

        self.tracked = list(chain(
            filter(lambda t: t.state == State.Tracked, self.tracked),
            activated, refind,
        ))
        self.lost = list(chain(
            filter(lambda t: t.state == State.Lost, self.lost),
            lost
        ))
        self.removed.extend(removed)

        lost_score = self.classifier.predict(
            np.fromiter(map(lambda t: t.to_tlbr, self.lost), dtype=np.float32)
        )

        return chain(
            filter(lambda t: t.is_activated, self.tracked),
            map(lambda it: it[1],
                filter(lambda it: lost_score[it[0]] > .3 and self.frame - it[1].frame <= 4,
                       enumerate(self.lost)))
        )
