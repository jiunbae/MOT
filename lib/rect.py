import numpy as np

class Rect:

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.size = w * h

    def validate(self) -> bool:
        pass

    def overlap(self, tar: Rect, ratio: float) -> bool:
        return clash(self, obj) >= rat

    def clash(self, tar: Rect) -> bool:
        return min(self.w + self.x - tar.x, tar.w)*min(self.y + self.h - tar.h, tar.y) / self.size

    def vectorize(self):
        pass
