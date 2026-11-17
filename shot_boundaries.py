"""Compare neighboring frames so stale local boxes stop at hard cuts."""

import cv2


class ShotBoundaries:
    def __init__(self):
        self._previous = None

    @staticmethod
    def _palette(signature):
        return cv2.calcHist([signature], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3)

    def advance(self, frame):
        current = cv2.resize(frame, (24, 14), interpolation=cv2.INTER_AREA)
        previous, self._previous = self._previous, current
        if previous is None:
            return False

        difference = sum(cv2.mean(cv2.absdiff(previous, current))[:3])
        if difference > 120:
            return True
        if difference < 65:
            return False
        # A palette check also catches some similar-luminance cuts without
        # treating ordinary object motion as a different shot.
        color_change = cv2.compareHist(self._palette(previous), self._palette(current),
                                      cv2.HISTCMP_BHATTACHARYYA)
        return color_change >= 0.22
