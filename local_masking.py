"""Local body-region masking for the timed preview.

Detection only returns local rectangles. Misses, uncertain detections and
tracking loss cannot switch the output to a full-screen cover.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from detector import EXPLICIT, EXPOSED, FrameGuard


MODEL_SIZE = 320



# CLIP verifies a proposed crop against clothing, normal skin and scenery.
# The low-light fallback also requires a localized hip proposal; venue
# classification only activates the additional search and cannot add a mask.
_STAGE_CONTEXT = (
    "an adult pole dancing show with a dancer in a thong performing on a pole in a strip club",
    "a woman pole dancing on an adult nightclub stage while a man throws cash",
    "a striptease dancer and a dance pole in an adult cabaret",
    "a crowd in ordinary clothes dancing under nightclub lights",
    "fully clothed people socializing in a nightclub, concert or theater",
    "a pool party, people in swimwear, pool and palm trees",
    "people wearing regular clothes during a conversation indoors",
    "the city street or a stage with a band or clothed audience",
)
_CLOSE_UNDERWEAR_CONTEXT = (
    "a close view of a woman wearing a bra and her bare midriff",
    "a woman in her underwear, wearing a bra and panties in an interior",
    "a close view of a woman's sports bra, shirt or crop top",
    "a close view of a man's bare chest or a clothed person",
    "a close view of arms, furniture, objects or a table, hands or products",
    "a person present but obscured by an object or people",
)
_UNDERWEAR_OUTFIT = (
    "a woman wearing only lingerie, a bra and panties",
    "a woman in a bra and underwear with bare thighs in an indoor room",
    "a woman wearing a bra and underpants, her midriff and thighs are exposed",
    "a woman standing and wearing just underwear",
    "a woman in ordinary clothes and denim shorts",
    "a clothed woman in a sports bra or tank top and short jeans",
    "a shirtless man wearing pants or shorts",
    "a woman wearing a bikini or swimsuit at a pool or beach",
    "a woman wearing a dress, bodysuit or athletic wear",
    "fully dressed people, a room, a wall or objects",
    "a man or woman with a bare chest",
    "a portrait of a person in normal clothes",
)
_UNDERWEAR_HIPS = (
    "a woman's hips in just panties and exposed buttocks",
    "a waist, buttocks and bare thighs in underpants",
    "a woman's hips covered only by a thin thong or panties",
    "an opaque pair of shorts, jeans, a skirt or dress",
    "an exposed upper arm, bare midriff or a bare chest",
    "the waist of a shirtless man in trousers",
    "bare legs or skin with no underwear visible",
    "a table, furniture or a dark background",
    "a woman's hips fully covered by a bodysuit",
)
_UNDERWEAR_TOP = (
    "a woman wearing just a bra above her exposed midriff",
    "a woman's breasts covered by a bra, a lingerie top",
    "a side view of an upper body wearing a woman's bra and bare midriff",
    "a shirt, a sleeveless top, sports bra, crop top or tank top",
    "an opaque dress covering torso or a clothed person's torso",
    "the bare chest of a man or shoulder",
    "a background, hands, forearm, or furniture",
)
_CROP_HIPS = (
    "bare exposed buttock cheeks and a thin g-string on a woman, viewed from behind",
    "a video game crop of exposed female buttock cheek on a pole dancer wearing a g-string",
    "bare hips and a woman's exposed buttocks seen from the side",
    "a person's hips covered by opaque jeans, trousers or shorts",
    "a woman's buttocks covered by opaque underwear, a skirt or a dress",
    "cloth covering the waist and hips of a person, clothing",
    "a frontal view of a woman wearing a bikini or top",
    "bare chest, torso, shoulders, face, arms or ordinary legs of a person",
    "dark foreground clothing and generic objects in the background",
    "a dancer's face, shoulders, bikini top or ordinary skin above the waist",
)
_PERSON_OUTFITS = (
    "a female dancer in a thin thong revealing her hips, full-body stage view",
    "a person with bare face and bare hips wearing a narrow thong or g-string",
    "woman on her toes holding a dance pole, uncovered buttocks in a thin g-string",
    "a person who is heavily underdressed in short pole dance costume",
    "a person in a long opaque bodysuit or catsuit that covers their entire waist and hips",
    "a woman wearing an opaque tight leather corset with shorts or leggings",
    "a fully clothed person wearing opaque trousers under dramatic colored lighting",
    "a person standing with a rifle and wearing a full-face ski mask and leggings",
)
_CROP_PARTS = (
    "bare exposed female breasts and nipples",
    "a woman's bare exposed buttocks",
    "uncovered human genitalia or anus",
    "a shirtless man with a bare chest and exposed nipples",
    "a person wearing a shirt, clothing, shorts or underwear",
    "a person wearing an opaque top covering the breasts",
    "a face, neck, arm, legs, belly or ordinary non-sensitive exposed skin",
    "a landscape, wall, street, object or background without nude body parts",
)


def _box_crop(frame, box, pad):
    height, width = frame.shape[:2]
    x, y, r, b = box
    dx, dy = round((r - x) * pad), round((b - y) * pad)
    return frame[max(0, y - dy):min(height, b + dy), max(0, x - dx):min(width, r + dx)]


class _ExposureCheck:
    """Re-use the installed local vision encoder for a bounded set of crops."""

    def __init__(self, session_options):
        import onnxruntime as ort
        from scene_guard import shared_scene_guard

        self.clip = shared_scene_guard()
        if not self.clip.available:
            raise RuntimeError(f"Body crop verifier unavailable: {self.clip.error}")
        torch, clip = self.clip._torch, self.clip._model
        tokenizer = self.clip._processor.tokenizer
        groups = (_STAGE_CONTEXT, _CROP_HIPS, _CROP_PARTS, _PERSON_OUTFITS,
                  _UNDERWEAR_OUTFIT, _UNDERWEAR_HIPS, _UNDERWEAR_TOP, _CLOSE_UNDERWEAR_CONTEXT)
        prompts = sum(groups, ())
        with torch.inference_mode():
            tokens = tokenizer(
                ["a video game screenshot of " + label for label in prompts],
                return_tensors="pt", padding=True,
            ).to(self.clip.device)
            vectors = clip.get_text_features(**tokens)
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        (self.context, self.hips, self.parts, self.outfits, self.underwear,
         self.underwear_hips, self.underwear_top, self.close_underwear) = vectors.split(
            [len(group) for group in groups], dim=0)
        # The app already installs this COCO detector for vehicle proposals.
        # Here we use only its person class to prioritize compact hip crops.
        model = Path(__file__).with_name("models") / "yolov5n-v7.onnx"
        self.underwear_regions = torch.cat((self.underwear_top, self.underwear_hips))
        self.people = (
            ort.InferenceSession(str(model), sess_options=session_options,
                                 providers=["CPUExecutionProvider"])
            if model.is_file() else None
        )

    def score(self, crops, reference):
        from PIL import Image, ImageOps

        torch, clip = self.clip._torch, self.clip._model
        processor = self.clip._processor.image_processor
        images = [
            ImageOps.pad(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                (224, 224), method=Image.Resampling.BILINEAR, color=(123, 117, 104),
            )
            for crop in crops
        ]
        pixels = processor(images=images, return_tensors="pt").pixel_values
        with torch.inference_mode():
            vectors = clip.get_image_features(pixel_values=pixels.to(self.clip.device))
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
            return (vectors @ reference.T).cpu().numpy()

    def stage_context(self, frame):
        # Activate extra searches only for underlit, colored footage. This is
        # independent of time, scene position, or a specific lighting hue.
        preview = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        _, saturation, value = cv2.split(cv2.cvtColor(preview, cv2.COLOR_BGR2HSV))
        dark = (value < 70).mean()
        colored = ((saturation > 125) & (value > 55)).mean()
        if dark < 0.69 or colored < 0.13:
            return False
        detailed = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
        scores = self.score((detailed,), self.context)[0]
        # Audience, tables and stage lights can score well even with a pole
        # dancer in frame. Require the performance cue to be competitive.
        return scores[:3].max() >= scores[3:].max() - 0.036

    def people_boxes(self, frame):
        if self.people is None:
            return []
        height, width = frame.shape[:2]
        inp = self.people.get_inputs()[0]
        size = inp.shape[-1] if isinstance(inp.shape[-1], int) else 640
        scale = size / max(width, height)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
        left, top = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
        image = cv2.copyMakeBorder(
            resized, top, size - resized.shape[0] - top, left,
            size - resized.shape[1] - left, cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        dtype = np.float16 if inp.type == "tensor(float16)" else np.float32
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, swapRB=True).astype(dtype, copy=False)
        rows = self.people.run(None, {inp.name: blob})[0][0].astype(np.float32, copy=False)
        confidence = rows[:, 4] * rows[:, 5]
        selected = np.flatnonzero(confidence >= 0.26)
        boxes = rows[selected, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] * 0.5
        survivors = cv2.dnn.NMSBoxes(boxes.tolist(), confidence[selected].tolist(), 0.26, 0.45)
        result = []
        for rank in np.asarray(survivors).reshape(-1):
            x, y, w, h = boxes[rank]
            x, y = round((x - left) / scale), round((y - top) / scale)
            w, h = round(w / scale), round(h / scale)
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            w, h = x2 - x1, y2 - y1
            # A cropped shoulder at the edge is not a reliable whole-person
            # anchor. Part proposals can still find an exposed area at an edge.
            visible_person = x1 > 0.015 * width and x2 < 0.985 * width
            if visible_person and h >= 0.36 * height and w >= 0.09 * height and h >= 1.10 * w:
                result.append((x1, y1, x2, y2))
            if len(result) >= 4:
                break
        return result


class LocalMaskDetector:
    """Propose regions with NudeNet, then verify that the crop shows exposure."""

    def __init__(self, *, detector=None, threads=3):
        self.detector = detector
        self.verifier = None
        self.reset()
        if detector is None:
            import nudenet
            import onnxruntime as ort
            from nudenet.nudenet import _postprocess, _read_image

            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(threads))
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            model = Path(nudenet.__file__).with_name("320n.onnx")
            self.session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.read_image = _read_image
            self.postprocess = _postprocess
            self.verifier = _ExposureCheck(options)
            self.verifier.score((np.zeros((32, 32, 3), dtype=np.uint8),), self.verifier.parts)

    def reset(self):
        self.stage_confirmed = False
        self.previous_hips = []
        self.previous_gray = None
        self.previous_torsos = []
        self.previous_torso_gray = None
        self.previous_garments = []
        self.previous_garment_gray = None
        self.previous_parts = []
        self.previous_part_gray = None
        self.garment_samples = 0

    @staticmethod
    def _flow_box(previous_gray, gray, prior, size):
        x, y, r, b = (n // 2 for n in prior)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y:b, x:r] = 255
        start = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=24, qualityLevel=0.025,
            minDistance=4, mask=mask, blockSize=5,
        )
        if start is None or len(start) < 6:
            return None
        end, valid, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, start, None, winSize=(19, 19), maxLevel=2,
        )
        if end is None:
            return None
        reverse, valid_reverse, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous_gray, end, None, winSize=(19, 19), maxLevel=2,
        )
        if reverse is None:
            return None
        stable = (
            valid.ravel().astype(bool) & valid_reverse.ravel().astype(bool)
            & (np.linalg.norm((reverse - start)[:, 0], axis=-1) < 1.6)
        )
        if stable.sum() < max(6, len(start) // 2):
            return None
        motion = (end - start)[stable, 0, :]
        dx, dy = np.median(motion, axis=0)
        scatter = np.linalg.norm(motion - (dx, dy), axis=-1)
        if np.median(scatter) > 2.0 or abs(dx) > (r - x) * 0.28 or abs(dy) > (b - y) * 0.28:
            return None
        x, y, r, b = prior
        height, width = size
        sx, sy = round(float(dx) * 2), round(float(dy) * 2)
        clipped = max(0, x + sx), max(0, y + sy), min(width, r + sx), min(height, b + sy)
        return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None

    @classmethod
    def _carry_boxes(cls, frame, boxes, previous, previous_gray, *, max_age, hold_age):
        """Carry confirmed local boxes briefly through weak measurements."""
        if not boxes and not previous:
            return [], None
        height, width = frame.shape[:2]
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
        current = [(box, 0) for box in boxes]
        if previous_gray is not None and gray.shape == previous_gray.shape:
            for prior, age in previous:
                if age >= max_age or any(_iou(prior, box) >= 0.20 for box, _ in current):
                    continue
                box = cls._flow_box(previous_gray, gray, prior, (height, width))
                if box is None and age < hold_age:
                    box = prior
                if box is not None and not any(_iou(box, newer) > 0.25 for newer, _ in current):
                    current.append((box, age + 1))
        return current, gray

    @staticmethod
    def _stage_torso(hip, person, shape):
        """Position the upper-body mosaic on the owner of a confirmed hip."""
        height, width = shape[:2]
        x, y, r, b = hip
        w, h = r - x, b - y
        if person is not None:
            px, py, pr, pb = person
            ph = pb - py
            left, right = px, pr
            top = py + round(ph * 0.16)
            bottom = max(py + round(ph * 0.43), y + round(h * 0.10))
            if top >= y or bottom - top > ph * 0.52:
                person = None
        if person is None:
            left, right = x - round(w * 0.12), r + round(w * 0.12)
            top, bottom = y - round(h * 1.45), y + round(h * 0.12)
        bottom = min(height, max(0, bottom))
        return max(0, left), max(0, top), min(width, right), bottom

    def _stage_boxes(self, frame, proposals):
        hips = [box for box, _ in proposals]
        torsos = [self._stage_torso(box, person, frame.shape) for box, person in proposals]
        torsos = [box for box in torsos if box[2] > box[0] and box[3] > box[1]]
        self.previous_hips, self.previous_gray = self._carry_boxes(
            frame, hips, self.previous_hips, self.previous_gray, max_age=11, hold_age=10)
        self.previous_torsos, self.previous_torso_gray = self._carry_boxes(
            frame, torsos, self.previous_torsos, self.previous_torso_gray, max_age=11, hold_age=10)
        return [(box, 0.24) for box, _ in self.previous_hips] + [
            (box, 0.18) for box, _ in self.previous_torsos]

    @staticmethod
    def _garment_region(box, top, bottom, *, inset=0.04):
        x, y, r, b = box
        w, h = r - x, b - y
        return x + round(w * inset), y + round(h * top), r - round(w * inset), y + round(h * bottom)

    def _worn_underwear(self, frame, people):
        if not people:
            return []
        verify = self.verifier
        outfits = verify.score([_box_crop(frame, box, 0.08) for box in people], verify.underwear)
        owners = []
        for box, scores in zip(people, outfits):
            positive = float(scores[:4].max())
            margin = positive - float(scores[4:].max())
            if positive >= 0.249 and margin >= 0.004:
                owners.append((box, positive, margin))
        if not owners:
            return []
        tops = [self._garment_region(box, 0.18, 0.47) for box, _, _ in owners]
        hips = [self._garment_region(box, 0.49, 0.77) for box, _, _ in owners]
        regions = tops + hips
        measurements = verify.score([_box_crop(frame, box, 0.08) for box in regions], verify.underwear_regions)
        top_scores = measurements[:len(owners), :len(_UNDERWEAR_TOP)]
        hip_scores = measurements[len(owners):, len(_UNDERWEAR_TOP):]
        selected = []
        for (person, outfit, margin), top, hip in zip(owners, top_scores, hip_scores):
            top_score, hip_score = float(top[:3].max()), float(hip[:3].max())
            above = top_score - float(top[3:].max())
            below = hip_score - float(hip[3:].max())
            # A neighboring person's shorts or ordinary bare legs can make
            # a loose subject crop look like underwear. Require local evidence.
            has_hips = (
                (outfit >= 0.272 and hip_score >= 0.270 and below >= 0.004)
                or (outfit >= 0.300 and hip_score >= 0.305 and below >= -0.003)
                or (margin >= 0.012 and hip_score >= 0.305 and below >= 0.0015)
                or (hip_score >= 0.280 and below >= 0.013)
            )
            has_top = top_score >= 0.255 and (
                (has_hips and above >= -0.003)
                or (outfit >= 0.280 and margin >= 0.012 and above >= 0.008 and below >= -0.006)
            )
            if has_top:
                selected.append(self._garment_region(person, 0.23, 0.44, inset=0.10))
            if has_hips:
                selected.append(self._garment_region(person, 0.48, 0.72, inset=0.08))
        return selected

    def _cropped_underwear_top(self, frame):
        """Localize a bra when close framing removes the person's face and legs."""
        verify = self.verifier
        small = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
        scene = verify.score((small,), verify.close_underwear)[0]
        if scene[:2].max() < 0.25 or scene[:2].max() < scene[2:].max() - 0.020:
            return []
        height, width = frame.shape[:2]
        tile_width, tile_height = min(width, round(height * 0.66)), round(height * 0.70)
        if min(tile_width, tile_height) < 24:
            return []
        left_edges = sorted({round(x) for x in np.linspace(0, width - tile_width, num=5)})
        proposals = []
        for left in left_edges:
            tile = frame[:tile_height, left:left + tile_width]
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)
            raw = self._network_output(enhanced)[0][0].T
            confidences = raw[:, 20]  # NudeNet FEMALE_BREAST_COVERED.
            index = int(confidences.argmax())
            strength = float(confidences[index])
            if strength < 0.12:
                continue
            cx, cy, w, h = raw[index, :4] * (max(tile_width, tile_height) / MODEL_SIZE)
            if not (0.09 * height < w < 0.36 * height and 0.09 * height < h < 0.42 * height):
                continue
            box = (max(0, left + round(cx - w / 2)), max(0, round(cy - h / 2)),
                   min(width, left + round(cx + w / 2)), min(height, round(cy + h / 2)))
            if min(box[2] - box[0], box[3] - box[1]) > 0:
                proposals.append((strength, box))
        unique = []
        for strength, box in sorted(proposals, reverse=True):
            if not any(_iou(box, prior) > 0.30 for _, prior in unique):
                unique.append((strength, box))
        if not unique:
            return []
        scores = verify.score([_box_crop(frame, box, 0.40) for _, box in unique], verify.underwear_top)
        return [box for (strength, box), values in zip(unique, scores)
                if values[:3].max() >= 0.25
                and values[:3].max() - values[3:].max() >= (-0.006 if strength >= 0.23 else 0.003)]

    def _underwear_boxes(self, frame):
        candidates = []
        if self.garment_samples % 2 == 0:
            people = self.verifier.people_boxes(frame)
            candidates = self._worn_underwear(frame, people)
            if not people:
                candidates = self._cropped_underwear_top(frame)
        self.garment_samples += 1
        self.previous_garments, self.previous_garment_gray = self._carry_boxes(
            frame, candidates, self.previous_garments, self.previous_garment_gray,
            max_age=7, hold_age=4)
        return [box for box, _ in self.previous_garments]

    def _network_output(self, frame):

        height, width = frame.shape[:2]
        side = max(width, height)
        step, remainder = divmod(side, MODEL_SIZE)
        if step and not remainder and width % step == height % step == 0:
            # NudeNet zero-pads the source to a square before its linear resize.
            # For integer sampling strides, resizing first is pixel-equivalent.
            # Its two channel swaps cancel for an already-BGR input.
            small = cv2.resize(frame, (width // step, height // step), interpolation=cv2.INTER_LINEAR)
            square = cv2.copyMakeBorder(
                small, 0, MODEL_SIZE - small.shape[0], 0, MODEL_SIZE - small.shape[1],
                cv2.BORDER_CONSTANT,
            )
            blob = cv2.dnn.blobFromImage(square, 1 / 255.0)
            x_pad, y_pad = side - width, side - height
            x_ratio, y_ratio = side / width, side / height
        else:
            blob, x_ratio, y_ratio, x_pad, y_pad, _, _ = self.read_image(frame, MODEL_SIZE)

        output = self.session.run(None, {self.input_name: blob})[0]
        return output, x_ratio, y_ratio, x_pad, y_pad

    def _infer(self, frame):
        if self.detector is not None:
            return self.detector.detect(frame)
        height, width = frame.shape[:2]
        output, x_ratio, y_ratio, x_pad, y_pad = self._network_output(frame)
        # Keep the original ordering for NudeNet's built-in NMS. Below-0.2
        # candidates would be discarded by the stock decoder's Python loop.
        candidates = np.flatnonzero(output[0, 4:, :].max(axis=0) >= 0.2)
        if not candidates.size:
            return []
        if candidates.size == 1:
            # The stock decoder squeezes singleton axes, so retain one row it
            # will ignore. No box or score is invented.
            keep = int(candidates[0])
            candidates = [0, keep] if keep else [0, 1]
        return self.postprocess(
            [output[:, :, candidates]], x_pad, y_pad, x_ratio, y_ratio, width, height,
            MODEL_SIZE, MODEL_SIZE,
        )

    def _hip_boxes(self, frame):
        """Multi-scale hip proposals for color-cast, low-light views."""
        height, width = frame.shape[:2]
        people = self.verifier.people_boxes(frame)
        outfit_margins = []
        if people:
            outfits = self.verifier.score([_box_crop(frame, box, 0.10) for box in people],
                                         self.verifier.outfits)
            # Verify the whole subject, since tight opaque fabric at the hips
            # can resemble bare skin in an isolated and color-cast crop.
            outfit_margins = outfits[:, :4].max(axis=1) - outfits[:, 4:].max(axis=1)
        proposals = []
        # Overlapping square crops preserve small anatomy in wide shots.
        side = min(width, height)
        left_edges = sorted({0, (width - side) // 2, width - side})
        areas = [(left, 0, left + side, height) for left in left_edges]
        if self.previous_hips:
            # Re-read the area around a confirmed moving target at a closer
            # scale. One bounded crop helps when the person crouches or the
            # light changes, without broadening the mask to the whole person.
            (x, y, r, b), _ = self.previous_hips[0]
            w, h = r - x, b - y
            wide = max(w, h)
            areas.append((max(0, x - wide), max(0, y - wide),
                          min(width, r + wide), min(height, b + wide)))
        for left, top, right, bottom in areas:
            tile = frame[top:bottom, left:right]
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR)
            raw = self._network_output(enhanced)[0][0].T
            scores = np.maximum(raw[:, 6], raw[:, 21])  # exposed and covered buttocks
            selected = np.flatnonzero(scores >= 0.12)
            for index in selected[np.argsort(scores[selected])[-5:]]:
                scale = max(tile.shape[:2]) / MODEL_SIZE
                center_x, center_y, w, h = raw[index, :4] * scale
                center_x += left
                center_y += top
                if not (0.045 * height < w < 0.40 * height and 0.035 * height < h < 0.24 * height):
                    continue
                if not (top <= center_y < bottom and left < center_x < right):
                    continue
                box = (
                    max(0, round(center_x - w / 2)), max(0, round(center_y - h / 2)),
                    min(width, round(center_x + w / 2)), min(height, round(center_y + h / 2)),
                )
                owner = next(
                    (i for i, (x, y, r, b) in enumerate(people)
                     if x <= center_x <= r and y + 0.40 * (b - y) <= center_y <= b), None,
                )
                if owner is not None or scores[index] >= 0.25:
                    proposals.append((float(scores[index]), box, owner))
        direct = []
        for proposal in sorted(proposals, reverse=True):
            if not any(_iou(proposal[1], other[1]) > 0.50 for other in direct):
                direct.append(proposal)
            if len(direct) >= 4:
                break
        proposals = direct
        # Pose-sized hip windows provide another scale when no specific hip
        # label survives. Candidate position comes from the detected person.
        for owner, (x, y, r, b) in enumerate(people):
            w, h = r - x, b - y
            for lo, hi in ((0.10, 0.85), (0.25, 0.95)):
                for vertical in (0.42, 0.55):
                    box = (
                        x + round(lo * w), y + round(vertical * h),
                        x + round(hi * w), y + round((vertical + 0.28) * h),
                    )
                    if min(box[2] - box[0], box[3] - box[1]) >= 20:
                        proposals.append((0., box, owner))
        if not proposals:
            return []
        scores = self.verifier.score(
            [_box_crop(frame, box, 0.18) for _, box, _ in proposals],
            self.verifier.hips,
        )
        selected = []
        for (direct_score, box, owner), values in zip(proposals, scores):
            exposed = float(values[:3].max())
            margin = exposed - float(values[3:].max())
            linked = any(age <= 2 and _iou(box, prior) >= 0.25
                         for prior, age in self.previous_hips)
            threshold = 0.012 if direct_score >= 0.16 else 0.018
            if owner is not None:
                context_confirmed = outfit_margins[owner] >= (0.004 if linked or direct_score >= 0.20 else 0.016)
                if not context_confirmed:
                    continue
            elif direct_score < 0.25:
                continue
            if linked:
                # A matching local proposal has information from the current
                # frame as well as a previously verified patch.
                threshold -= 0.009
            if exposed < 0.25 or margin < threshold:
                continue
            strength = margin + 0.08 * min(direct_score, 0.35)
            selected.append((strength + (0.006 if linked else 0.), box, owner, direct_score))
        result, owners = [], set()
        for strength, box, owner, direct_score in sorted(selected, key=lambda proposal: proposal[0], reverse=True):
            if result and (direct_score < 0.24 or strength < 0.035):
                continue
            if owner in owners or any(_iou(box, other) > 0.25 for other, _ in result):
                continue
            if owner is not None:
                owners.add(owner)
            result.append((box, people[owner] if owner is not None else None))
            if len(result) >= 2:
                break
        return result

    def detect(self, frame):
        """Return padded xyxy rectangles. The right and bottom edges are exclusive."""
        if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
                or min(frame.shape[:2]) < 2):
            raise ValueError("Local masking requires an 8-bit BGR frame.")
        height, width = frame.shape[:2]
        candidates = []
        for item in self._infer(frame):
            label = item["class"]
            score = float(item["score"])
            if label in EXPLICIT:
                threshold = 0.48
                group = 2
            elif label == "BUTTOCKS_EXPOSED":
                threshold = 0.50
                group = 1
            elif label in EXPOSED:
                threshold = 0.45
                group = 0
            else:
                continue
            if not math.isfinite(score) or score < threshold:
                continue
            x, y, w, h = (int(value) for value in item["box"])
            box = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((group, score, box))

        selected = []
        if candidates and self.verifier is not None:
            candidates = sorted(candidates, key=lambda value: value[1], reverse=True)[:6]
            scores = self.verifier.score(
                [_box_crop(frame, box, 0.60) for _, _, box in candidates],
                self.verifier.parts,
            )
            for (group, _, box), result in zip(candidates, scores):
                if result[group] >= result[3:].max() + 0.010:
                    selected.append(box)
        elif self.detector is not None:
            return [self._expand(box, 0.35, width, height) for _, _, box in candidates]

        self.previous_parts, self.previous_part_gray = self._carry_boxes(
            frame, selected, self.previous_parts, self.previous_part_gray, max_age=7, hold_age=4)
        selected = [(box, 0.33) for box, _ in self.previous_parts]

        if self.verifier is not None and (self.stage_confirmed or self.verifier.stage_context(frame)):
            # A verified stage is the same environment until the input cuts.
            # A brief foreground occlusion should not turn off the local search.
            hips = self._hip_boxes(frame)
            self.stage_confirmed = self.stage_confirmed or bool(hips)
            for box, padding in self._stage_boxes(frame, hips):
                if not any(_iou(box, earlier) > 0.30 for earlier, _ in selected):
                    selected.append((box, padding))
            self.previous_garments = []
            self.previous_garment_gray = None
        elif self.verifier is not None:
            self.previous_hips = []
            self.previous_gray = None
            self.previous_torsos = []
            self.previous_torso_gray = None
            for box in self._underwear_boxes(frame):
                if not any(_iou(box, earlier) > 0.30 for earlier, _ in selected):
                    selected.append((box, 0.15))

        return [self._expand(box, padding, width, height) for box, padding in selected]

    @staticmethod
    def _expand(box, padding, width, height):
        x, y, r, b = box
        dx, dy = max(4, round((r - x) * padding)), max(4, round((b - y) * padding))
        return max(0, x - dx), max(0, y - dy), min(width, r + dx), min(height, b + dy)


def _iou(first, second):
    ax, ay, ar, ab = first
    bx, by, br, bb = second
    overlap = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    if not overlap:
        return 0.0
    return overlap / ((ar - ax) * (ab - ay) + (br - bx) * (bb - by) - overlap)


@dataclass(frozen=True)
class _Track:
    box: tuple
    seconds: float
    velocity: tuple = (0.0, 0.0)

    def at(self, seconds, horizon):
        dt = min(horizon, max(0.0, seconds - self.seconds))
        dx, dy = (round(value * dt) for value in self.velocity)
        x1, y1, x2, y2 = self.box
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy


@dataclass(frozen=True)
class MaskSnapshot:
    """An immutable detector result evaluated later against an output timestamp."""

    tracks: tuple
    ttl_seconds: float
    prediction_seconds: float

    def active(self, seconds):
        return [
            track.at(seconds, self.prediction_seconds) for track in self.tracks
            if track.seconds <= seconds and seconds - track.seconds <= self.ttl_seconds
        ]


class MaskTracker:
    """Carry a local mask briefly across detector samples on the same shot.

    Frames and detections must use the same media clock and size. Call clear()
    on a seek or shot cut. Update and active belong to the presentation owner.
    """

    def __init__(self, *, ttl_seconds=0.24, prediction_seconds=0.12, retain_unmatched=True):
        if not 0 < prediction_seconds <= ttl_seconds:
            raise ValueError("Mask prediction must be shorter than retention.")
        self.ttl = ttl_seconds
        self.prediction = prediction_seconds
        self.retain_unmatched = retain_unmatched
        self.clear()

    def clear(self):
        self.last_update = -math.inf
        self.tracks = []

    def update(self, boxes, seconds):
        if seconds <= self.last_update:
            return
        self.last_update = seconds
        unmatched = [track for track in self.tracks if seconds - track.seconds <= self.ttl]
        updated = []
        for box in boxes:
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            scores = [_iou(box, track.at(seconds, self.prediction)) for track in unmatched]
            index = max(range(len(scores)), key=scores.__getitem__, default=None)
            if index is None or scores[index] < 0.12:
                updated.append(_Track(box, seconds))
                continue
            previous = unmatched.pop(index)
            dt = seconds - previous.seconds
            # Track center motion, not growth induced by unstable detector boxes.
            dx = (box[0] + box[2] - previous.box[0] - previous.box[2]) * 0.5 / dt
            dy = (box[1] + box[3] - previous.box[1] - previous.box[3]) * 0.5 / dt
            vx = 0.65 * dx + 0.35 * previous.velocity[0]
            vy = 0.65 * dy + 0.35 * previous.velocity[1]
            updated.append(_Track(box, seconds, (vx, vy)))
        # Detectors that already handle continuity can publish an authoritative
        # set, preventing a second tracker from prolonging rejected masks.
        self.tracks = updated + unmatched if self.retain_unmatched else updated

    def snapshot(self):
        return MaskSnapshot(tuple(self.tracks), self.ttl, self.prediction)

    def active(self, seconds):
        return self.snapshot().active(seconds)


def _union_area(boxes):
    edges = sorted({x for x1, _, x2, _ in boxes for x in (x1, x2)})
    area = 0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((top, bottom) for low, top, high, bottom in boxes
                           if low < right and high > left)
        first = last = 0
        covered = 0
        for top, bottom in intervals:
            if top > last:
                covered += last - first
                first, last = top, bottom
            else:
                last = max(last, bottom)
        area += (right - left) * (covered + last - first)
    return area


def bounded_boxes(frame_shape, *groups):
    """Keep local masks within 65% combined coverage, favoring earlier groups."""
    height, width = frame_shape[:2]
    area_limit = height * width * 0.65
    occupied = []
    covered_area = 0
    result = []
    for group in groups:
        current = []
        for x1, y1, x2, y2 in group:
            clipped = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            low, top, high, bottom = clipped
            if high <= low or bottom <= top or clipped in occupied:
                continue
            new_area = _union_area([*occupied, clipped])
            if new_area > area_limit:
                # Keep the center of a close-up inside a local mask. Even if
                # it does not overlap any other mask, the cropped box fits.
                w, h = high - low, bottom - top
                scale = math.sqrt(max(0, area_limit - covered_area) / (w * h))
                cropped_w, cropped_h = int(w * scale), int(h * scale)
                if min(cropped_w, cropped_h) < 2:
                    continue
                left, upper = low + (w - cropped_w) // 2, top + (h - cropped_h) // 2
                clipped = left, upper, left + cropped_w, upper + cropped_h
                new_area = _union_area([*occupied, clipped])
            if new_area <= covered_area:
                continue
            occupied.append(clipped)
            covered_area = new_area
            current.append(clipped)
        result.append(tuple(current))
    return tuple(result)


def mosaic(frame, boxes, *, solid=False, blocks=3):
    """Render local masks in place; all pixels outside the clipped boxes stay unchanged."""
    height, width = frame.shape[:2]
    if blocks < 1:
        raise ValueError("A mosaic needs at least one block per axis.")
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if solid or min(h, w) < 12:
            roi[:] = (44, 42, 40)
            continue
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_AREA)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame


class ColorFilter:
    """The existing red-spot filter at a bounded analysis resolution.

    This returns the input unchanged when no local red filter applies.
    Large red areas never produce a full-screen cover.
    """

    def __init__(self, mode="selective", *, sample_size=448):
        if mode not in ("off", "selective", "always"):
            raise ValueError(f"Unknown color filter mode: {mode}")
        if sample_size < 32:
            raise ValueError("Color analysis resolution is too small.")
        self.mode = mode
        self.sample_size = sample_size
        # Only the color heuristic is used here. Supplying a detector keeps
        # the independent region model out of the color-filter setup.
        self.locator = FrameGuard(detector=self, scene_mode="basic", blood_mode="selective")

    def apply(self, frame):
        if self.mode == "off":
            return frame, False
        if self.mode == "always":
            return self.locator._grayscale(frame), True

        height, width = frame.shape[:2]
        scale = min(1.0, self.sample_size / max(height, width))
        if scale < 1.0:
            size = max(1, round(width * scale)), max(1, round(height * scale))
            sample = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        else:
            sample = frame
        filtered, changed, _ = self.locator._filter_red(sample)
        if not changed:
            return frame, False
        if scale == 1.0:
            return filtered, True

        # Re-check the selected pixels at output size so an upscaled mask
        # cannot desaturate adjacent non-red texture.
        delta = cv2.absdiff(sample, filtered)
        marked = cv2.bitwise_not(cv2.inRange(delta, (0, 0, 0), (0, 0, 0)))
        pixels = cv2.resize(marked, (width, height), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.divmod(np.flatnonzero(pixels), width)
        raw = frame[ys, xs]
        red, _ = self.locator._red_channels(raw)
        if not red.any():
            return frame, False
        bgr = raw[red].astype(np.int32)
        grey = (29 * bgr[:, 0] + 150 * bgr[:, 1] + 77 * bgr[:, 2]) >> 8
        result = frame.copy()
        result[ys[red], xs[red]] = grey[:, None]
        return result, True
