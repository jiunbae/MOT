from typing import Iterable
from itertools import chain, compress

import numpy as np

from models.identifier import Identifier
from models.classifier import Classifier
from lib import trace, matching
from utils.nms import nms
from utils.kalmanfilter import KalmanFilter


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

        self.identifier = Identifier().load()
        self.classifier = Classifier().load()

        self.frame = 0

    def update(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray) \
            -> Iterable[trace.Trace]:
        self.frame += 1

        refind, lost = [], []
        activated, removed = [], []
        # Step 1. Prediction
        for track in chain(self.tracked, self.lost):
            track.predict()

        # Step 2. Selection by score
        if scores is None:
            scores = np.ones(np.size(boxes, 0), dtype=float)

        detections = list(chain(
            map(lambda t: trace.Trace(*t, from_det=True), zip(boxes, scores)),
            map(lambda t: trace.Trace(*t, from_det=False), zip(boxes, scores))
        ))

        self.classifier.update(image)

        detections.extend(map(lambda t: trace.Trace(t.tracking(image), t.track_score, from_det=True),
                              filter(lambda t: t.is_activated, chain(self.tracked, self.lost))))

        rois = np.asarray(list(map(lambda t: t.to_tlbr, detections)), np.float32)

        class_scores = self.classifier.predict(rois)
        scores = np.concatenate([
            np.ones(np.size(boxes, 0), dtype=np.float32),
            np.fromiter(map(lambda t: t.score, detections[np.size(boxes, 0):]), dtype=np.float32)
        ]) * class_scores

        # Non-maxima suppression
        if len(detections) > 0:
            mask = np.zeros(np.size(rois, 0), dtype=np.bool)
            mask[list(nms(rois, scores.reshape(-1), threshold=.4))] = True

            indices = np.zeros_like(detections, dtype=np.bool)
            indices[np.where(mask & (scores >= self.min_score))] = True

            detections = list(compress(detections, indices))
            scores = scores[indices]

            for detection, score in zip(detections, scores):
                detection.score = score

        predictions = list(filter(lambda t: not t.from_det, detections))
        detections = list(filter(lambda t: t.from_det, detections))

        # set features
        features = self.identifier.extract(image, np.asarray(
            list(map(lambda t: t.to_tlbr, detections)), dtype=np.float32)
        )

        for idx, detection in enumerate(detections):
            detection.feature = features[idx]

        # Step3. Association for tracked
        # matching for tracked target
        unconfirmed = list(filter(lambda t: not t.is_activated, self.tracked))
        tracked = list(filter(lambda t: t.is_activated, self.tracked))

        distance = matching.nearest_distance(tracked, detections, metric='euclidean')
        cost = matching.gate_cost(self.motion, distance, tracked, detections)
        matches, u_track, u_detection = matching.assignment(cost, threshold=self.min_dist)

        for track, det in matches:
            tracked[track].update(self.frame, image, detections[det])

        # matching for missing targets
        detections = list(map(lambda u: detections[u], u_detection))
        distance = matching.nearest_distance(self.lost, detections, metric='euclidean')
        cost = matching.gate_cost(self.motion, distance, self.lost, detections)
        matches, u_lost, u_detection = matching.assignment(cost, threshold=self.min_dist)

        for miss, det in matches:
            self.lost[miss].reactivate(self.frame, image, detections[det], reassign=not self.use_refind)
            refind.append(self.lost[miss])

        # remaining tracked
        matched_size = len(u_detection)
        detections = list(map(lambda u: detections[u], u_detection)) + predictions
        u_tracked = list(map(lambda u: tracked[u], u_track))
        distance = matching.iou_distance(u_tracked, detections)
        matches, u_track, u_detection = matching.assignment(distance, threshold=.8)

        for track, det in matches:
            u_tracked[track].update(self.frame, image, detections[det], update_feature=True)

        for track in map(lambda u: u_tracked[u], u_track):
            track.lost()
            lost.append(track)

        # unconfirmed
        detections = list(map(lambda u: detections[u], filter(lambda u: u < matched_size, u_detection)))
        distance = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.assignment(distance, threshold=.8)

        for track, det in matches:
            unconfirmed[track].update(self.frame, image, detections[det], update_feature=True)

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
            filter(lambda t: t.state == trace.State.Tracked, self.tracked),
            activated, refind,
        ))
        self.lost = list(chain(
            filter(lambda t: t.state == trace.State.Lost, self.lost),
            lost
        ))
        self.removed.extend(removed)

        lost_score = self.classifier.predict(
            np.asarray(list(map(lambda t: t.to_tlbr, self.lost)), dtype=np.float32)
        )

        return chain(
            filter(lambda t: t.is_activated, self.tracked),
            map(lambda it: it[1],
                filter(lambda it: lost_score[it[0]] > .3 and self.frame - it[1].frame <= 4,
                       enumerate(self.lost)))
        )
