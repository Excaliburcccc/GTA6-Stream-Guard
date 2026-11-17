"""Detect and track marked police cars for local mosaic.

The object proposal model is Ultralytics YOLOv5n v7.0 (COCO), from
https://github.com/ultralytics/yolov5/releases/tag/v7.0. The existing CLIP
encoder distinguishes marked patrol vehicles from ordinary vehicles by crop.
All inference stays local. Boxes use (left, top, right, bottom) pixel corners.
"""
from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolov5n-v7.onnx"
MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
MODEL_SHA256 = "04f0e55c26f58d17145b36045780fe1250d5bd2187543e11568e5141d05b3262"
VEHICLE_CLASSES = (2, 7)  # COCO car and truck.
PATROL = (
    "a marked police car with a visible police roof light bar",
    "a black and white police cruiser with POLICE lettering",
    "a police SUV with blue and red flashing emergency lights",
    "a marked law enforcement patrol vehicle",
    "a damaged marked police car in a rollover crash with emergency lights and a marked roof number",
    "an official police patrol car with a narrow roof-mounted light bar and red or blue flashing beacons",
    "a police patrol car with a black-and-white paint scheme and large POLICE lettering",
    "an emergency police vehicle with a marked unit number and red or blue flashing police lights",
    "a crash-damaged police car with POLICE lettering or a unit number on its roof",
)
OTHER = (
    "an ordinary civilian passenger car or SUV",
    "a civilian sedan or sports car driving on a city street",
    "a taxi or a civilian delivery car with a rooftop sign",
    "an ambulance, fire truck, or other emergency vehicle that is not a police car",
    "a civilian car with an armed passenger leaning out of an open window",
    "a modified racing car with racing sponsor markings or underbody neon lights",
    "an ordinary tuned street racing car at a car meet with a large decorative race number, decals and spectators",
    "a track race car seen from behind, with a rear spoiler and a black and white racing number",
    "a shiny stock dark gray or black SUV or pickup, with headlight and taillight glare",
    "a plain cargo delivery or work truck with commercial branding, logos, graphics or graffiti",
    "a civilian car in a parking garage, race track, airport or nightclub, with colorful reflections",
    "a building, street lamp, wall, road, signs, trees, people, motorcycle or rider",
)

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Candidate:
    box: Box
    confidence: float
    vehicle_is_primary: bool = True


@dataclass(slots=True)
class Track:
    corners: np.ndarray
    box: Box
    proof_at: int
    confirmed_at: int
    verified_at: int
    matched_at: int
    reliable: bool = True
    anchor: np.ndarray | None = None
    unseen: int = 0

    @property
    def is_confirmed(self):
        return self.confirmed_at >= 0


@dataclass(frozen=True, slots=True)
class PatrolCue:
    candidate: Candidate
    margin: float
    eligible: bool
    instant: bool
    confirming: bool


def _overlap(first, second):
    ax, ay, ar, ab = first
    bx, by, br, bb = second
    intersection = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    union = (ar - ax) * (ab - ay) + (br - bx) * (bb - by) - intersection
    return intersection / union if union else 0.0


def _expand(box, width, height, extra_x, extra_y):
    x1, y1, x2, y2 = box
    dx = round((x2 - x1) * extra_x)
    dy = round((y2 - y1) * extra_y)
    return max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)


def _box_corners(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _bounds(corners, width, height):
    low = corners.min(axis=0)
    high = corners.max(axis=0)
    return (
        max(0, math.floor(float(low[0]))), max(0, math.floor(float(low[1]))),
        min(width, math.ceil(float(high[0]))), min(height, math.ceil(float(high[1]))),
    )


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _clip_polygon(corners, width, height):
    clipped = corners.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    return clipped


def _beacon_components(binary):
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    height, width = binary.shape
    return [
        (float(cx), float(cy), int(w), int(h))
        for (x, y, w, h, area), (cx, cy) in zip(stats[1:count], centers[1:count], strict=True)
        if 2 <= area <= width * height * 0.06 and w <= width * 0.62 and h <= height * 0.25
    ]


def _lighting(frame, box):
    # Compare light sources on the vehicle, not just any two colors in its crop.
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if min(w, h) < 15:
        return 0, False
    roof = frame[y1:y1 + round(h * 0.68), x1:x2]
    if w > 210:
        roof = cv2.resize(roof, (210, max(1, round(roof.shape[0] * 210 / w))), interpolation=cv2.INTER_AREA)
    b, g, r = roof.astype(np.int16).transpose(2, 0, 1)
    blue = _beacon_components((b >= 114) & (b - r >= 37) & (b - g >= 25))
    red = _beacon_components((r >= 145) & (r - b >= 43) & (r - g >= 52))
    ch, cw = b.shape
    paired = any(
        abs(yb - yr) <= max(3.0, ch * 0.17)
        and cw * 0.06 <= abs(xb - xr) <= cw * 0.72
        and cw * 0.27 <= (xb + xr) * 0.5 <= cw * 0.73
        for xb, yb, _, _ in blue for xr, yr, _, _ in red
    )

    # A grille or roof strip has more than one illuminated pixel horizontally
    # and is not the isolated taillight at a lower outer corner.
    def central_strip(components):
        return any(
            cw * 0.16 <= x <= cw * 0.84
            and (y < ch * 0.55 or cw * 0.28 <= x <= cw * 0.72)
            and width >= max(3, cw * 0.022) and width >= height
            for x, y, width, height in components
        )

    colors = int(central_strip(blue)) | (int(central_strip(red)) << 1)
    return colors, paired


class PoliceVehicleDetector:
    """Find patrol cars on the analysis thread, never on the preview thread."""

    def __init__(self, scene_guard=None, model_path=MODEL_PATH):
        self.available = False
        self.error = ""
        self.reset()
        try:
            model_path = Path(model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"Missing patrol proposal model: {model_path}")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 3
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self._session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
            self._size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
            if scene_guard is None or not scene_guard.available or not hasattr(scene_guard, "_model"):
                from scene_guard import shared_scene_guard
                scene_guard = shared_scene_guard()
            if not scene_guard.available:
                raise RuntimeError(f"Patrol classifier unavailable: {scene_guard.error}")
            self._clip = scene_guard
            self._torch = scene_guard._torch
            self._processor = scene_guard._processor
            self._normalize = self._torch.nn.functional.normalize
            text = tuple(f"a video game screenshot cropped to {s}" for s in PATROL + OTHER)
            tokens = self._processor.tokenizer(text, padding=True, return_tensors="pt")
            with self._torch.inference_mode():
                features = scene_guard._model.get_text_features(**tokens.to(scene_guard.device))
                self._text = self._normalize(features.float(), dim=-1)
            self.available = True
        except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def reset(self):
        self._tracks: list[Track] = []
        self._tick = 0
        self._last_extra_search = -4
        self._previous_gray = None

    def _proposals(self, frame):
        height, width = frame.shape[:2]
        side = self._size
        scale = min(side / width, side / height)
        scaled_width, scaled_height = max(1, round(width * scale)), max(1, round(height * scale))
        x_offset = (side - scaled_width) // 2
        y_offset = (side - scaled_height) // 2
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        canvas[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = cv2.resize(
            frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
        )
        inputs = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        predictions = self._session.run(None, {self._input_name: inputs.astype(self._input_dtype, copy=False)})[0][0].astype(np.float32, copy=False)
        scores = predictions[:, 4] * predictions[:, [5 + cls for cls in VEHICLE_CLASSES]].max(axis=1)
        dominant = np.isin(predictions[:, 5:].argmax(axis=1), VEHICLE_CLASSES)
        keep = np.flatnonzero(scores >= 0.16)
        if not keep.size:
            return []
        rows = predictions[keep, :4]
        boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2,
                                 rows[:, 2], rows[:, 3]))
        selected = cv2.dnn.NMSBoxes(boxes.tolist(), scores[keep].tolist(), 0.16, 0.45)
        selected = np.asarray(selected, dtype=np.int32).ravel()
        selected = sorted(selected, key=lambda j: scores[keep[j]], reverse=True)[:14]
        candidates = []
        for j in selected:
            x, y, w, h = boxes[j]
            x1 = max(0, round((x - x_offset) / scale))
            y1 = max(0, round((y - y_offset) / scale))
            x2 = min(width, round((x + w - x_offset) / scale))
            y2 = min(height, round((y + h - y_offset) / scale))
            if x2 - x1 >= 16 and y2 - y1 >= 12:
                candidates.append(Candidate((x1, y1, x2, y2), float(scores[keep[j]]), bool(dominant[keep[j]])))
        return candidates

    def _classify(self, frame, candidates):
        from PIL import Image, ImageOps

        height, width = frame.shape[:2]
        size = self._clip._model.config.vision_config.image_size
        crops = []
        for candidate in candidates:
            x1, y1, x2, y2 = _expand(candidate.box, width, height, 0.06, 0.15)
            rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append(ImageOps.pad(
                Image.fromarray(rgb), (size, size), method=Image.Resampling.BICUBIC,
                color=(123, 117, 104)
            ))
        pixels = self._processor.image_processor(crops, return_tensors="pt").pixel_values
        with self._torch.inference_mode():
            features = self._clip._model.get_image_features(pixel_values=pixels.to(self._clip.device))
            sims = (self._normalize(features.float(), dim=-1) @ self._text.T).cpu().numpy()
        marked = sims[:, :len(PATROL)].max(axis=1)
        other = sims[:, len(PATROL):].max(axis=1)
        return list(zip(marked.tolist(), (marked - other).tolist(), strict=True))

    def _gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.shape[1] > 640:
            gray = cv2.resize(gray, (640, round(frame.shape[0] * 640 / frame.shape[1])),
                              interpolation=cv2.INTER_AREA)
        return gray

    def _track_shape(self, track, gray, width, height):
        previous = track.anchor if track.anchor is not None else self._previous_gray
        if previous is None or previous.shape != gray.shape:
            return None
        ratio = np.array([gray.shape[1] / width, gray.shape[0] / height], dtype=np.float32)
        points_of_box = track.corners * ratio
        left, top, right, bottom = _bounds(points_of_box, gray.shape[1], gray.shape[0])
        if min(right - left, bottom - top) < 13:
            return None
        center = points_of_box.mean(axis=0)
        interior = center + (points_of_box - center) * 0.85
        mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(interior - (left, top)).astype(np.int32), 255)
        points = cv2.goodFeaturesToTrack(
            previous[top:bottom, left:right], 80, 0.009, 3, mask=mask
        )
        if points is None or len(points) < 6:
            return None
        points += np.array([left, top], dtype=np.float32)
        lk = dict(winSize=(35, 35), maxLevel=4)
        moved, available, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **lk)
        if moved is None or available is None:
            return None
        finite = available.ravel().astype(bool) & np.isfinite(moved).all(axis=(1, 2))
        if finite.sum() < 6:
            return None
        source, target = points[finite], moved[finite]
        returned, good, _ = cv2.calcOpticalFlowPyrLK(gray, previous, target, None, **lk)
        if returned is None or good is None:
            return None
        agrees = good.ravel().astype(bool) & (np.linalg.norm(
            (returned - source)[:, 0, :], axis=1
        ) <= 3.0)
        if agrees.sum() < 6:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            source[agrees], target[agrees], method=cv2.RANSAC,
            ransacReprojThreshold=3.3, maxIters=140
        )
        if matrix is None or inliers is None:
            return None
        selected = inliers.ravel().astype(bool)
        if selected.sum() < max(6, 0.37 * len(points)):
            return None
        positions = source[agrees][selected, 0, :]
        if np.min(np.ptp(positions, axis=0)) < 8:
            return None
        scale = np.linalg.norm(matrix[:, 0])
        if not 0.59 <= scale <= 1.60:
            return None
        transformed = cv2.transform(points_of_box[None], matrix)[0] / ratio
        forecast = _bounds(transformed, width, height)
        area = _area(forecast)
        if not (0.17 * _area(track.box) <= area <= 2.8 * _area(track.box) and 0 < area < width * height * 0.60):
            return None
        return transformed

    def _match_rotating_features(self, track, gray, width, height):
        # Short-range matching of points inside the known vehicle survives
        # larger rotations than frame-to-frame optical flow.
        previous = track.anchor if track.anchor is not None else self._previous_gray
        if previous is None or previous.shape != gray.shape:
            return None
        ratio = np.array([gray.shape[1] / width, gray.shape[0] / height], dtype=np.float32)
        corners = track.corners * ratio
        screen_height, screen_width = gray.shape
        left, top, right, bottom = _bounds(corners, screen_width, screen_height)
        if min(right - left, bottom - top) < 34:
            return None
        dx = round((right - left) * 0.52)
        dy = round((bottom - top) * 0.58)
        x1, y1 = max(0, left - dx), max(0, top - dy)
        x2, y2 = min(screen_width, right + dx), min(screen_height, bottom + dy)
        mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        center = corners.mean(axis=0)
        inside = center + (corners - center) * 0.85
        cv2.fillConvexPoly(mask, np.rint(inside - (left, top)).astype(np.int32), 255)
        sift = cv2.SIFT_create(nfeatures=100, contrastThreshold=0.022)
        key1, desc1 = sift.detectAndCompute(previous[top:bottom, left:right], mask)
        key2, desc2 = sift.detectAndCompute(gray[y1:y2, x1:x2], None)
        if desc1 is None or desc2 is None or min(len(key1), len(key2)) < 5:
            return None
        bf = cv2.BFMatcher(cv2.NORM_L2)
        forward = bf.knnMatch(desc1, desc2, k=2)
        backward = {item.queryIdx: item.trainIdx for item in bf.match(desc2, desc1)}
        paired = [
            near for group in forward if len(group) == 2 for near, far in [group]
            if near.distance < 0.76 * far.distance and backward.get(near.trainIdx) == near.queryIdx
        ]
        if len(paired) < 5:
            return None
        source = np.float32([key1[m.queryIdx].pt for m in paired]) + (left, top)
        target = np.float32([key2[m.trainIdx].pt for m in paired]) + (x1, y1)
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=140
        )
        if matrix is None or inliers is None:
            return None
        hits = inliers.ravel().astype(bool)
        if hits.sum() < 5 or np.min(np.ptp(source[hits], axis=0)) < 11:
            return None
        scale = np.linalg.norm(matrix[:, 0])
        if not 0.58 <= scale <= 1.55:
            return None
        found = cv2.transform(corners[None], matrix)[0] / ratio
        area = _area(_bounds(found, width, height))
        if not 0.21 * _area(track.box) <= area <= 2.8 * _area(track.box):
            return None
        return found

    def _recover_turned_vehicle(self, frame, track, forecast):
        """Search the immediate neighborhood of one known car in rotated views."""
        height, width = frame.shape[:2]
        bx = _bounds(forecast, width, height) if forecast is not None else track.box
        x1, y1, x2, y2 = _expand(bx, width, height, 0.51, 0.43)
        view = frame[y1:y2, x1:x2]
        h, w = view.shape[:2]
        if min(w, h) < 24:
            return None
        tilt = 0.0
        if forecast is not None:
            dx, dy = forecast[1] - forecast[0]
            tilt = math.degrees(math.atan2(dy, dx))
        rotations = (
            (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE)
            if tilt >= 0 else (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)
        )
        for turn in rotations:
            upright = cv2.rotate(view, turn)
            selected = []
            for obj in self._proposals(upright):
                if obj.confidence < 0.27 or not obj.vehicle_is_primary:
                    continue
                u, v, p, q = obj.box
                if turn == cv2.ROTATE_180:
                    restored = w - p, h - q, w - u, h - v
                elif turn == cv2.ROTATE_90_CLOCKWISE:
                    restored = v, h - p, q, h - u
                else:
                    restored = w - q, u, w - v, p
                a, b, c, d = restored
                original = a + x1, b + y1, c + x1, d + y1
                overlap = _overlap(original, bx)
                ratio = _area(original) / max(1, _area(bx))
                minimum_iou = 0.18 if forecast is not None else 0.22
                if overlap >= minimum_iou and 0.36 <= ratio <= 2.6:
                    selected.append((overlap * obj.confidence, obj, original))
            if not selected:
                continue
            best = max(selected, key=lambda item: item[0])
            obj, original = best[1:]
            score, margin = self._classify(upright, [obj])[0]
            if score >= 0.23 and (margin >= -0.017 or (forecast is not None and best[0] > 0.22)):
                return original
        return None

    def _cues(self, frame, candidates):
        sides = self._classify(frame, candidates) if candidates else []
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        b, g, r = small.astype(np.int16).transpose(2, 0, 1)
        vibrant = (((b >= 90) & (b - g > 32) & (b - r > -30))
                   | ((r >= 100) & (r - g > 35) & (r - b > 28)))
        # In colored ambient lighting, project a higher evidence threshold
        # on reflected colors and favor independent markings.
        ambient = np.median((30 * r + 59 * g + 11 * b) / 100) < 75 and vibrant.mean() > 0.048
        cues = []
        for candidate, (marked, gap) in zip(candidates, sides, strict=True):
            colors, paired = _lighting(frame, candidate.box)
            plausible = candidate.vehicle_is_primary and candidate.confidence >= 0.22
            marked_body = marked >= 0.297 and gap >= 0.061 and candidate.confidence >= 0.39
            marked_lamp = marked >= 0.248 and gap >= (0.031 if ambient else 0.007) and bool(colors)
            dual_lamp = marked >= 0.248 and gap >= (0.031 if ambient else 0.008) and paired
            x1, y1, x2, y2 = candidate.box
            height, width = frame.shape[:2]
            partial_front = (
                not ambient and marked >= 0.300 and gap >= 0.014 and colors == 3
                and (x1 <= width * 0.025 or x2 >= width * 0.975)
                and y2 >= height * 0.74 and x2 - x1 >= width * 0.18
            )
            eligible = plausible and (marked_body or marked_lamp or dual_lamp or partial_front)
            instant = plausible and (
                partial_front or (
                    paired and marked >= 0.266 and gap >= (0.039 if ambient else 0.022)
                    and (colors == 3 or gap >= 0.030)
                )
                or (marked >= 0.31 and gap >= 0.075 and candidate.confidence >= 0.48)
            )
            # Two weak lighting observations can both be ordinary reflections.
            # Keep weak cues for following a known car, but require a stronger
            # light or livery observation before placing the first mosaic.
            confirming = (
                marked_body or (paired and (gap >= 0.016 or (colors == 3 and gap >= 0.010)))
                or (bool(colors & 1) and gap >= 0.015)
                or (bool(colors & 2) and gap >= 0.023)
            )
            cues.append(PatrolCue(candidate, float(gap), eligible, instant, confirming))
        return cues

    def _masks(self, frame, tracks):
        height, width = frame.shape[:2]
        # A confirmed car can survive three temporarily untrackable frames.
        # The shot boundary still clears every track before drawing again.
        return [
            _expand(track.box, width, height, 0.13, 0.18)
            for track in tracks if track.is_confirmed and (
                track.reliable or (track.unseen <= 3 and _area(track.box) >= width * height * 0.004)
            )
        ]

    def detect(self, frame_bgr) -> list[Box]:
        if not self.available:
            return []
        self._tick += 1
        tick = self._tick
        height, width = frame_bgr.shape[:2]
        cues = self._cues(frame_bgr, self._proposals(frame_bgr))
        if not cues and not self._tracks:
            self._previous_gray = None
            return []
        gray = self._gray(frame_bgr) if self._tracks or any(cue.eligible for cue in cues) else None
        forecasts = []
        for track in self._tracks:
            # Fresh, well-overlapping detections make optical flow unnecessary.
            exact = any(cue.eligible and _overlap(cue.candidate.box, track.box) > 0.52 for cue in cues)
            can_follow = track.is_confirmed and track.unseen <= 3
            forecast = None if exact or not can_follow else self._track_shape(track, gray, width, height)
            if forecast is None and can_follow and not exact:
                if _area(track.box) >= width * height * 0.021 and tick - track.matched_at <= 7:
                    forecast = self._match_rotating_features(track, gray, width, height)
            forecasts.append(forecast)
        pairs = []
        for j, track in enumerate(self._tracks):
            forecast = forecasts[j]
            predicted = _bounds(forecast, width, height) if forecast is not None else track.box
            for i, cue in enumerate(cues):
                box = cue.candidate.box
                overlap = _overlap(box, predicted)
                if not cue.eligible and (not track.is_confirmed or cue.margin < -0.031):
                    continue
                required_iou = 0.20 if forecast is not None else 0.33
                if overlap >= required_iou:
                    pairs.append((overlap + 0.07 * cue.eligible, i, j))
        used_cues, used_tracks = set(), set()
        current = []
        for _, i, j in sorted(pairs, reverse=True):
            if i in used_cues or j in used_tracks:
                continue
            cue, track = cues[i], self._tracks[j]
            if not cue.eligible and not cue.candidate.vehicle_is_primary:
                continue
            # Geometry alone cannot keep a track alive indefinitely on a
            # civilian vehicle crossing the old location.
            if not cue.eligible and tick - track.verified_at > 12:
                continue
            used_cues.add(i)
            used_tracks.add(j)
            box = cue.candidate.box
            verified_now = cue.eligible and cue.confirming
            confirmed = track.confirmed_at
            if confirmed < 0 and (
                cue.instant or (verified_now and track.proof_at >= 0 and tick - track.proof_at <= 3)
            ):
                confirmed = tick
            current.append(Track(
                _box_corners(box), box, tick if verified_now else track.proof_at, confirmed,
                tick if cue.eligible else track.verified_at, tick
            ))
        rescue_used = False
        ranked_old = sorted(enumerate(self._tracks), key=lambda item: _area(item[1].box), reverse=True)
        for j, track in ranked_old:
            if j in used_tracks or not track.is_confirmed or tick - track.verified_at > 12:
                continue
            forecast = forecasts[j]
            predicted = _bounds(forecast, width, height) if forecast is not None else track.box
            if any(_overlap(predicted, picked.box) > 0.35 for picked in current):
                continue
            rescued = None
            if (not rescue_used and _area(track.box) >= frame_bgr.shape[0] * frame_bgr.shape[1] * 0.019
                    and tick - track.matched_at <= 6 and (forecast is None or tick - track.matched_at >= 2)):
                rescue_used = True
                rescued = self._recover_turned_vehicle(frame_bgr, track, forecast)
            if rescued is not None:
                current.append(Track(
                    _box_corners(rescued), rescued, track.proof_at,
                    track.confirmed_at, track.verified_at, tick
                ))
            elif forecast is not None and tick - track.matched_at <= 7:
                # Hold the original oriented corners, not an ever-growing
                # new axis-aligned rectangle after each roll.
                current.append(Track(
                    forecast, predicted, track.proof_at,
                    track.confirmed_at, track.verified_at, track.matched_at
                ))
            elif track.unseen <= 3 and tick - track.matched_at <= 4:
                # Keep a latent identity for reacquisition, but no frozen
                # mosaic when neither the image nor a detector supports it.
                current.append(Track(
                    track.corners, track.box, track.proof_at,
                    track.confirmed_at, track.verified_at, track.matched_at,
                    reliable=False, anchor=track.anchor, unseen=track.unseen + 1
                ))
        for i, cue in enumerate(cues):
            if i in used_cues or not cue.eligible:
                continue
            box = cue.candidate.box
            if any(_overlap(box, picked.box) >= 0.31 for picked in current):
                continue
            current.append(Track(
                _box_corners(box), box, tick if cue.confirming else -1, tick if cue.instant else -1, tick, tick
            ))
        for track in current:
            if track.is_confirmed and track.reliable:
                track.anchor = gray
        self._tracks = current
        self._previous_gray = gray if current else None
        return self._masks(frame_bgr, current)

    def advance(self, frame_bgr) -> list[Box]:
        """Follow established cars between samples without running either detector."""
        if not self.available or not any(track.is_confirmed for track in self._tracks):
            return []
        height, width = frame_bgr.shape[:2]
        gray = self._gray(frame_bgr)
        for track in self._tracks:
            if not track.is_confirmed or track.unseen > 3:
                continue
            updated = self._track_shape(track, gray, width, height)
            if (updated is None and _area(track.box) >= width * height * 0.021
                    and self._tick - track.matched_at <= 4):
                updated = self._match_rotating_features(track, gray, width, height)
            if updated is None:
                track.reliable = False
                track.unseen += 1
            else:
                track.reliable = True
                track.unseen = 0
                track.anchor = gray
                track.corners = updated
                track.box = _bounds(updated, width, height)
        self._previous_gray = gray
        if self._tick - self._last_extra_search >= 2 and any(
            track.is_confirmed and track.unseen == 1
            and self._tick - track.verified_at <= 2
            and _area(track.box) >= width * height * 0.055
            for track in self._tracks
        ):
            self._last_extra_search = self._tick + 1
            return self.detect(frame_bgr)
        return self._masks(frame_bgr, self._tracks)
