import numpy as np

from itertools import chain, groupby

from lib.trace import State, Trace

from models.classification.classifier import PatchClassifier
from models.reid import load_reid_model, extract_reid_features


class Tracker(object):
    def __init__(self,
                 min_score: float = .2, min_dist: float = .64, min_time: int = 120):

        self.min_score = min_score
        self.min_dist = min_dist
        self.min_time = min_time

        self.tracked = []
        self.lost = []

        self.classifier = PatchClassifier()
        self.identifier = load_reid_model()

        self.frame = 0

    def update(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray):
        self.frame += 1

        for track in chain(self.tracked, self.lost):
            track.predict()

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

        # TODO: nms

        predictions = filter(lambda t: not t.from_det, detections)
        detections = filter(lambda t: t.from_det, detections)

        features = extract_reid_features(self.identifier, image, map(lambda t: t.to_tlbr, detections))
        features = features.cpu().numpy()

        for idx, detection in enumerate(detections):
            detection.feature = features[1]

        unconfirmed, tracked = [], []

        group_key = lambda trace: trace.is_activated
        groupby(sorted(tracked, key=group_key), key=group_key)

        for track in self.tracked:

            tee(l)

            """step 3: association for tracked"""
            # matching for tracked targets
            unconfirmed = []
            tracked_stracks = []  # type: list[STrack]
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)

            dists = matching.nearest_reid_distance(tracked_stracks, detections, metric='euclidean')
            dists = matching.gate_cost_matrix(self.kalman_filter, dists, tracked_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)

            for itracked, idet in matches:
                tracked_stracks[itracked].update(detections[idet], self.frame_id, image)

            # matching for missing targets
            detections = [detections[i] for i in u_detection]
            dists = matching.nearest_reid_distance(self.lost_stracks, detections, metric='euclidean')
            dists = matching.gate_cost_matrix(self.kalman_filter, dists, self.lost_stracks, detections)
            matches, u_lost, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)
            for ilost, idet in matches:
                track = self.lost_stracks[ilost]  # type: STrack
                det = detections[idet]
                track.re_activate(det, self.frame_id, image, new_id=not self.use_refind)
                refind_stracks.append(track)

            # remaining tracked
            # tracked
            len_det = len(u_detection)
            detections = [detections[i] for i in u_detection] + pred_dets
            r_tracked_stracks = [tracked_stracks[i] for i in u_track]
            dists = matching.iou_distance(r_tracked_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.8)
            for itracked, idet in matches:
                r_tracked_stracks[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
            for it in u_track:
                track = r_tracked_stracks[it]
                track.lost()
                lost_stracks.append(track)

            # unconfirmed
            detections = [detections[i] for i in u_detection if i < len_det]
            dists = matching.iou_distance(unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.remove()
                removed_stracks.append(track)


class OnlineTracker(object):

    def __init__(self, min_cls_score=0.2, min_ap_dist=0.64, max_time_lost=120, use_tracking=True, use_refind=True):

        self.min_cls_score = min_cls_score
        self.min_ap_dist = min_ap_dist
        self.max_time_lost = max_time_lost

        self.kalman_filter = KalmanFilter()

        self.tracked_stracks = []   # type: list[STrack]
        self.lost_stracks = []      # type: list[STrack]
        self.removed_stracks = []   # type: list[STrack]

        self.use_refind = use_refind
        self.use_tracking = use_tracking
        self.classifier = PatchClassifier()
        self.reid_model = load_reid_model()

        self.frame_id = 0

    def update(self, image, tlwhs, det_scores=None):
        self.frame_id += 1

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        iou_tracking = []
        iou_finished = []

        """step 1: prediction"""
        for strack in itertools.chain(self.tracked_stracks, self.lost_stracks):
            strack.predict()

        """step 2: scoring and selection"""
        if det_scores is None:
            det_scores = np.ones(len(tlwhs), dtype=float)

        dets = tlwhs[np.where(det_scores>=.0)]
        updated_iou_track = []
        for track in iou_tracking:
            if len(dets) > 0:
                best = max(dets, key=lambda x: iou(track['bboxes'][-1], x))
                if iou(track, best) >= .3:
                    track['bboxes'].append(best)
                    track['score'] = max(track['score'], det_scores[np.where(tlwhs == best)])
                    updated_iou_track.append(track)
                    dets = np.delete(dets[np.where(dets == best)])
            if len(updated_iou_track) == 0 or track is not updated_iou_track[-1]:
                if track['score'] >= .65 and len(track['bboxes']) >= 3:
                    iou_finished.append(track)

        detections = []
        new_tracks = []
        for tlwh, score in zip(tlwhs, det_scores):
            detections.append(STrack(tlwh, score, from_det=True))
            new_tracks.append({'bboxes': [tlwh], 'score': score, 'start': self.frame_id})

        for tracks in updated_iou_track + new_tracks:
            detections.append(STrack(tracks['bboxes'][-1], tracks['score'], from_det=False))

        if self.classifier is None:
            pred_dets = []
        else:
            self.classifier.update(image)

            n_dets = len(tlwhs)
            if self.use_tracking:
                tracks = [STrack(t.self_tracking(image), t.tracklet_score(), from_det=False)
                          for t in itertools.chain(self.tracked_stracks, self.lost_stracks) if t.is_activated]
                detections.extend(tracks)
            rois = np.asarray([d.tlbr for d in detections], dtype=np.float32)

            cls_scores = self.classifier.predict(rois)
            scores = np.asarray([d.score for d in detections], dtype=np.float)
            scores[0:n_dets] = 1.
            scores = scores * cls_scores
            # nms
            if len(detections) > 0:
                keep = nms_detections(rois, scores.reshape(-1), nms_thresh=0.4)
                mask = np.zeros(len(rois), dtype=np.bool)
                mask[keep] = True
                keep = np.where(mask & (scores >= self.min_cls_score))[0]
                detections = [detections[i] for i in keep]
                scores = scores[keep]
                for d, score in zip(detections, scores):
                    d.score = score
            pred_dets = [d for d in detections if not d.from_det]
            detections = [d for d in detections if d.from_det]

        # set features
        tlbrs = [det.tlbr for det in detections]
        features = extract_reid_features(self.reid_model, image, tlbrs)
        features = features.cpu().numpy()
        for i, det in enumerate(detections):
            det.set_feature(features[i])

        """step 3: association for tracked"""
        # matching for tracked targets
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        dists = matching.nearest_reid_distance(tracked_stracks, detections, metric='euclidean')
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)

        for itracked, idet in matches:
            tracked_stracks[itracked].update(detections[idet], self.frame_id, image)

        # matching for missing targets
        detections = [detections[i] for i in u_detection]
        dists = matching.nearest_reid_distance(self.lost_stracks, detections, metric='euclidean')
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, self.lost_stracks, detections)
        matches, u_lost, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)
        for ilost, idet in matches:
            track = self.lost_stracks[ilost]  # type: STrack
            det = detections[idet]
            track.re_activate(det, self.frame_id, image, new_id=not self.use_refind)
            refind_stracks.append(track)

        # remaining tracked
        # tracked
        len_det = len(u_detection)
        detections = [detections[i] for i in u_detection] + pred_dets
        r_tracked_stracks = [tracked_stracks[i] for i in u_track]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.8)
        for itracked, idet in matches:
            r_tracked_stracks[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
        for it in u_track:
            track = r_tracked_stracks[it]
            track.lost()
            lost_stracks.append(track)

        # unconfirmed
        detections = [detections[i] for i in u_detection if i < len_det]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.remove()
            removed_stracks.append(track)

        """step 4: init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if not track.from_det or track.score < 0.6:
                continue
            track.activate(self.kalman_filter, self.frame_id, image)
            activated_starcks.append(track)

        """step 6: update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.remove()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == State.Tracked]
        self.lost_stracks = [t for t in self.lost_stracks if t.state == State.Lost]  # type: list[STrack]
        self.tracked_stracks.extend(activated_starcks)
        self.tracked_stracks.extend(refind_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # output_stracks = self.tracked_stracks + self.lost_stracks

        # get scores of lost tracks
        rois = np.asarray([t.tlbr for t in self.lost_stracks], dtype=np.float32)
        lost_cls_scores = self.classifier.predict(rois)
        out_lost_stracks = [t for i, t in enumerate(self.lost_stracks)
                            if lost_cls_scores[i] > 0.3 and self.frame_id - t.end_frame <= 4]
        output_tracked_stracks = [track for track in self.tracked_stracks if track.is_activated]

        output_stracks = output_tracked_stracks + out_lost_stracks

        return output_stracks