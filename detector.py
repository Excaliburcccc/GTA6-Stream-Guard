"""Local frame moderation. Input and output are uint8 OpenCV BGR frames.

NudeNet supplies region detections. The optional colour heuristic looks for
small dark red marks with texture on a lighter neutral background. Scene
triage, when enabled, uses a separate model. Thresholds are prototype settings.
"""

import numpy as np


EXPLICIT = {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED"}
EXPOSED = {"FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED"}
COVERED = {"FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED", "FEMALE_GENITALIA_COVERED"}


def _overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


class FrameGuard:
    """Pass a detector with .detect(frame) to exercise the processing path offline."""

    def __init__(self, detector=None, cover_style="mosaic", blood_mode="always",
                 scene_mode="full", scene_guard=None):
        if cover_style not in {"mosaic", "solid"}:
            raise ValueError("cover_style must be mosaic or solid")
        if blood_mode not in {"always", "selective"}:
            raise ValueError("blood_mode must be always or selective")
        if scene_mode not in {"full", "basic"}:
            raise ValueError("scene_mode must be full or basic")
        self.cover_style = cover_style
        self.blood_mode = blood_mode
        self.startup_error = ""
        self.last_error = ""
        self.detector = detector
        if detector is None:
            try:
                import onnxruntime

                onnxruntime.disable_telemetry_events()
                from nudenet import NudeDetector

                self.detector = NudeDetector()
            except Exception as exc:
                self.startup_error = f"{type(exc).__name__}: {exc}"
        self.scene_guard = None
        if scene_mode == "full":
            try:
                if scene_guard is None:
                    from scene_guard import SceneGuard

                    scene_guard = SceneGuard()
                self.scene_guard = scene_guard
                if not scene_guard.available:
                    raise RuntimeError(scene_guard.error)
            except Exception as exc:
                self.startup_error = " / ".join(
                    filter(None, (self.startup_error, f"Scene model: {exc}"))
                )
        self._last_boxes = []
        self._scene_signature = None
        self._frame_shape = None
        self._hold_remaining = 0

    @staticmethod
    def safe_frame(frame, reason=""):
        # Use no input pixels, including in case of an invalid/corrupt source frame.
        shape = getattr(frame, "shape", ())
        size = shape[:2] if len(shape) >= 2 and min(shape[:2]) > 0 else (720, 1280)
        return np.full((*size, 3), (35, 32, 29), dtype=np.uint8)

    def _hold(self, frame, reason):
        return self.safe_frame(frame), "hold:" + reason

    def _boxes(self, frame, detections):
        height, width = frame.shape[:2]
        boxes = []
        for item in detections:
            label = item["class"]
            if label not in EXPLICIT | EXPOSED | COVERED:
                continue
            score = float(item["score"])
            if not np.isfinite(score):
                raise ValueError("invalid score from detector")
            if score < 0.25:
                continue
            x, y, w, h = map(int, item["box"])
            if w <= 0 or h <= 0 or x >= width or y >= height or x + w <= 0 or y + h <= 0:
                raise ValueError("unusable region from detector")
            area = min(w, width) * min(h, height) / (width * height)
            if label in EXPLICIT or (label in EXPOSED and area >= 0.09):
                return [], "explicit-or-closeup"
            # A covered-body label alone is common in ordinary clothing. Only
            # flag large covered regions as a possible close-up.
            if label in COVERED and area < 0.12:
                continue
            if label in COVERED:
                return [], "possible-closeup"
            dx = max(4, int(w * 0.40))
            dy = max(4, int(h * 0.40))
            boxes.append((max(0, x - dx), max(0, y - dy),
                          min(width, x + w + dx), min(height, y + h + dy)))
        if len(boxes) >= 4:
            return [], "multiple-regions"
        return boxes, None

    @staticmethod
    def _red_channels(bgr):
        blue, green, red = np.moveaxis(bgr.astype(np.int16), -1, 0)
        red_mask = (red >= 40) & (red - green > 24) & (red - blue > 20)
        red_mask &= (green * 10 < red * 7) & (blue * 10 < red * 8)
        return red_mask, red

    def _filter_red(self, frame):
        import cv2

        h, w = frame.shape[:2]
        # Estimate shapes on a 640px reference image; resample for both 4K and SD.
        scale = min(1.0, 640 / max(h, w))
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        sample = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        coloured, red = self._red_channels(sample)
        binary = coloured.astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return (None if self.blood_mode == "always" else frame.copy()), False, False
        grey = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        channel_span = sample.max(axis=2).astype(np.int16) - sample.min(axis=2)
        broad = stats[:, cv2.CC_STAT_AREA] > sh * sw * 0.005
        broad[0] = False
        near_broad = cv2.dilate(broad[labels].astype(np.uint8), np.ones((17, 17), np.uint8))
        candidates = []
        for i in range(1, count):
            x, y, bw, bh, area = map(int, stats[i])
            # Reject large or stripe-like red objects before looking at texture.
            if not 4 <= area <= 0.003 * sh * sw or max(bw, bh) > 4 * min(bw, bh):
                continue
            cx, cy = np.rint(centers[i]).astype(int)
            if near_broad[cy, cx]:
                continue
            x0, y0 = max(0, x - 3), max(0, y - 3)
            x1, y1 = min(sw, x + bw + 3), min(sh, y + bh + 3)
            roi = np.s_[y0:y1, x0:x1]
            mask = labels[roi] == i
            red_pixels = red[roi][mask]
            # Include small dark spots, but reject bright lights, uniform flat
            # logos and glossy highlights on broad coloured surfaces.
            if np.median(red_pixels) > 163 or red_pixels.min() > 125:
                continue
            if red_pixels.max() - red_pixels.min() < 14:
                continue
            ring = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            ring &= ~mask
            if ring.sum() < 6:
                continue
            dark = np.median(grey[roi][mask])
            background = (grey[roi] >= max(75, dark + 23)) & (channel_span[roi] < 58)
            light_ring = np.count_nonzero(background & ring) / np.count_nonzero(ring)
            if light_ring < 0.65:
                continue
            candidates.append((i, area, light_ring))
        accepted = []
        for i, area, light_ring in candidates:
            # A cluster of nearby droplets carries more evidence than one blob.
            # Isolated marks need a clearly lighter neutral surround.
            if light_ring >= 0.90 and area >= 12:
                accepted.append(i)
                continue
            if any(j != i and np.sum((centers[i] - centers[j]) ** 2) <= 34 ** 2
                   for j, _, _ in candidates):
                accepted.append(i)
        if not accepted:
            return (None if self.blood_mode == "always" else frame.copy()), False, False
        selected = np.zeros(count, np.uint8)
        selected[accepted] = 1
        marked = selected[labels]
        if np.count_nonzero(marked) > marked.size * 0.045:
            return None, False, True
        if self.blood_mode == "always":
            return None, False, False
        mask = cv2.resize(marked, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        output = frame.copy()
        # A mask selected at analysis resolution can catch adjacent non-red
        # pixels in 4K: check the actual colour before compositing.
        raw = frame[mask]
        eligible = self._red_channels(raw)[0]
        selected_pixels = raw[eligible].astype(np.int32)
        luminance = (29 * selected_pixels[:, 0] + 150 * selected_pixels[:, 1]
                     + 77 * selected_pixels[:, 2]) >> 8
        raw[eligible] = luminance[:, None]
        output[mask] = raw
        return output, bool(eligible.any()), False

    @staticmethod
    def _grayscale(frame):
        # Fast OpenCV implementation in production. Keep the detector injectable
        # in environments with just NumPy for deterministic frame-policy tests.
        try:
            import cv2
        except ImportError:
            channels = frame.astype(np.int32)
            grey = (29 * channels[..., 0] + 150 * channels[..., 1]
                    + 77 * channels[..., 2]) >> 8
            return np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2)
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def _cover(self, image, box):
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        if self.cover_style == "solid" or min(roi.shape[:2]) < 18:
            roi[:] = (44, 42, 40)
            return
        # Cap at 3 x 3 blocks regardless of output resolution. Each block is
        # represented by its mean, not an individual sample from the original.
        for rows in np.array_split(roi, 3, axis=0):
            for block in np.array_split(rows, 3, axis=1):
                block[:] = block.mean(axis=(0, 1))

    def process(self, frame):
        if self.startup_error or self.detector is None:
            return self._hold(frame, "detector-unavailable")
        try:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 2):
                raise ValueError("expected uint8 BGR video frame")
            height, width = frame.shape[:2]
            # A new shot invalidates positions and buffered colour decisions.
            signature = frame[::max(1, height // 20), ::max(1, width // 28)].astype(np.int16)
            changed = (self._frame_shape != frame.shape or self._scene_signature is None
                       or np.abs(signature - self._scene_signature).mean() > 38)
            if changed:
                if self._last_boxes:
                    self._hold_remaining = max(3, self._hold_remaining)
                self._last_boxes = []
            self._scene_signature = signature
            self._frame_shape = frame.shape
            if self.scene_guard is not None:
                decision = self.scene_guard.analyze(frame)
                if not decision.available or decision.block:
                    self.last_error = self.scene_guard.error if not decision.available else ""
                    self._hold_remaining = 8
                    return self._hold(frame, decision.category)
            detections = self.detector.detect(frame)
            if not isinstance(detections, (list, tuple)):
                raise ValueError("invalid result from detector")
            boxes, hold_reason = self._boxes(frame, detections)
            previous_boxes = self._last_boxes
            self._last_boxes = boxes
            # A vanishing or fast-moving detection can leave old coordinates
            # pointing at empty background. Hide the game for several frames.
            lost = any(not any(_overlap(old, box) for box in boxes) for old in previous_boxes)
            if hold_reason or lost:
                self._hold_remaining = 8
                return self._hold(frame, hold_reason or "track-lost")
            if self._hold_remaining > 0:
                self._hold_remaining -= 1
                return self._hold(frame, "settling")
            output, filtered, heavy = self._filter_red(frame)
            if heavy:
                self._hold_remaining = 8
                return self._hold(frame, "large-textured-red-area")
            if self.blood_mode == "always":
                output = self._grayscale(frame)
                filtered = True
            for box in previous_boxes + boxes:
                self._cover(output, box)
            actions = ["cover"] if boxes else []
            if filtered:
                actions.append("gray" if self.blood_mode == "always" else "desaturate")
            self.last_error = ""
            return output, "+".join(actions) or "normal"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._hold_remaining = 8
            return self._hold(frame, "processing-error")
