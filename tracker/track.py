from enum import Enum


class State(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Replaced = 4


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = State.New

    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def lost(self):
        self.state = State.Lost

    def remove(self):
        self.state = State.Removed

    def replace(self):
        self.state = State.Replaced
