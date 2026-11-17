"""Find substantial red stains that warrant a local mosaic."""

import cv2
import numpy as np


class BloodMosaicDetector:
    """Recognize large pools or clusters of stains at a bounded resolution.

    Color, texture, shape and surrounding pixels are evaluated together so
    an ordinary red surface or an overall warm cast is weak evidence.
    """

    def __init__(self, sample_width=448, hold_samples=3):
        self.sample_width = sample_width
        self.hold_samples = hold_samples
        self.close_kernel = np.ones((3, 3), np.uint8)
        self.ring_kernel = np.ones((11, 11), np.uint8)
        self.reset()

    def reset(self):
        self.shape = None
        self.tracks = []

    def detect(self, frame):
        if self.shape != frame.shape:
            self.reset()
            self.shape = frame.shape
        boxes = self._find_regions(frame)
        # Confirmed boxes survive brief loss behind an animation or moving
        # foreground character. The caller resets them across shot changes.
        carry = [
            (previous, age + 1) for previous, age in self.tracks
            if age < self.hold_samples and not any(_overlaps(previous, box) for box in boxes)
        ]
        self.tracks = [(box, 0) for box in boxes] + carry
        return [box for box, _ in self.tracks]

    def _find_regions(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, self.sample_width / max(height, width))
        sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
        sample = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        b, g, r = cv2.split(sample.astype(np.int16))
        likely = (r >= 53) & (r - g > 26) & (r - b > 16) & (g * 100 < r * 67) & (b * 100 < r * 83)
        pixels = sw * sh
        minimum_pool = max(24, round(pixels * 0.012))
        if np.count_nonzero(likely) < minimum_pool:
            return []

        hue, saturation, value = cv2.split(cv2.cvtColor(sample, cv2.COLOR_BGR2HSV))
        likely &= (hue <= 8) | (hue >= 165)
        connected = cv2.morphologyEx(likely.astype(np.uint8), cv2.MORPH_CLOSE, self.close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
        neutral = (saturation < 70) & (value > 91)
        pool_boxes, near_candidates = [], []
        for label in np.flatnonzero(stats[1:count, cv2.CC_STAT_AREA] >= max(12, pixels * 0.0005)) + 1:
            x, y, w, h, area = (int(n) for n in stats[label])
            if min(w, h) < 3 or w > sw * 0.88 or h > sh * 0.72:
                continue
            left, top, right, bottom = max(0, x - 5), max(0, y - 5), min(sw, x + w + 5), min(sh, y + h + 5)
            window = np.s_[top:bottom, left:right]
            component = labels[window] == label
            region = component & likely[window]
            red_values = r[window][region]
            if red_values.size < 12:
                continue
            green_values = g[window][region]
            blue_values = b[window][region]
            surround = cv2.dilate(component.astype(np.uint8), self.ring_kernel).astype(bool) & ~component
            light_edge = np.count_nonzero(neutral[window] & surround) / max(1, np.count_nonzero(surround))
            box = x, y, x + w, y + h
            # Paint and red lighting tend to lack either a saturated dark
            # core or a lighter neutral edge. Warm wood also fails the hue
            # and surrounding-neutral tests.
            red_core = np.mean((green_values * 10 < red_values * 3) & (blue_values * 10 < red_values * 4))
            if (area >= minimum_pool and min(w, h) >= 6 and 0.18 <= area / (w * h) <= 0.76
                    and red_core >= 0.28 and light_edge >= 0.24
                    and np.std(red_values) >= 12 and np.median(red_values) >= 77):
                pool_boxes.append(box)
            elif light_edge >= 0.27 and np.mean(red_values - green_values) >= 32:
                near_candidates.append(box)

        def output_box(box):
            x1, y1, x2, y2 = box
            pad_x = max(5, round((x2 - x1) * 0.08))
            pad_y = max(5, round((y2 - y1) * 0.13))
            return (max(0, round((x1 - pad_x) * width / sw)), max(0, round((y1 - pad_y) * height / sh)),
                    min(width, round((x2 + pad_x) * width / sw)), min(height, round((y2 + pad_y) * height / sh)))

        roots = list(pool_boxes)
        guard = max(7, round(sh * 0.19))
        for candidate in near_candidates:
            # Let nearby stains extend the verified pool mask, rather than
            # treating isolated dark reds elsewhere as gore.
            x1, y1, x2, y2 = candidate
            for i, (a, b, c, d) in enumerate(roots):
                if max(x1 - c, a - x2, y1 - d, b - y2) <= guard:
                    pool_boxes[i] = _union(candidate, pool_boxes[i])
        return [output_box(box) for box in pool_boxes]


def _union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _overlaps(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    minimum = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return overlap >= 0.30 * minimum
