"""Timestamped video playback with background detection and local mosaics."""

from collections import deque
from dataclasses import dataclass
import math
import queue
import sys
import threading
import time

import cv2
import numpy as np

from app import WINDOW, WindowsKeys, card, fit, fps_of
from local_masking import ColorFilter, LocalMaskDetector, MaskTracker, _iou, bounded_boxes, mosaic
from shot_boundaries import ShotBoundaries


def manual_mosaic(image):
    """Blend a coarse pixel mosaic over the already processed program frame."""
    height, width = image.shape[:2]
    cells = cv2.resize(image, (max(1, round(width / 32)), max(1, round(height / 32))),
                       interpolation=cv2.INTER_AREA)
    pixels = cv2.resize(cells, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.addWeighted(pixels, 0.95, image, 0.05, 0, dst=pixels)
    return pixels


@dataclass(frozen=True, slots=True)
class VideoFrame:
    index: int
    pts: float
    shot: int
    captured: float
    pixels: np.ndarray
    color_changed: bool = False


class MaskChannel:
    """Bounded and nonblocking work queue for one kind of local detection."""

    def __init__(self, name, factory, rate, delay, stopped, *, every_frame=False, bridge_gap=False):
        self.name = name
        self.rate = rate
        self.delay = delay
        self.every_frame = every_frame
        self.bridge_gap = bridge_gap
        self.ready = threading.Event()
        self.stopped = stopped
        self._factory = factory
        self._jobs = deque(maxlen=max(6, math.ceil(delay * rate) + 4))
        self._results = deque(maxlen=math.ceil(max(delay, 2.0) * rate) + 36)
        self._changed = threading.Condition()
        self._last_shot = None
        self._next_sample = 0.0
        self.samples = 0
        self.tracking_updates = 0
        self.dropped_work = 0
        self.failed = False
        self._thread = threading.Thread(
            target=self._run, name=f"detect-{name}", daemon=True
        )
        self._thread.start()

    def set_input_fps(self, fps):
        if not self.every_frame:
            return
        with self._changed:
            self._jobs = deque(maxlen=max(12, math.ceil(self.delay * fps) + 8))
            self._results = deque(maxlen=max(36, math.ceil(max(self.delay, 2.0) * fps) + 36))

    def submit(self, frame):
        if self.failed:
            return
        due = frame.shot != self._last_shot or frame.pts + 0.0005 >= self._next_sample
        if not due and not self.every_frame:
            return
        if due:
            self._last_shot = frame.shot
            self._next_sample = frame.pts + 1 / self.rate
        with self._changed:
            if len(self._jobs) == self._jobs.maxlen:
                self.dropped_work += 1
            self._jobs.append((frame, due))
            self._changed.notify()

    def boxes_for(self, frame):
        with self._changed:
            for sample_index, pts, shot, snapshot in reversed(self._results):
                if sample_index > frame.index:
                    continue
                if shot != frame.shot or frame.pts - pts > 0.32:
                    return ()
                current = snapshot.active(frame.pts)
                if current or not self.bridge_gap:
                    return current
                return self._bridge_region_sample(frame)
        return ()

    def _bridge_region_sample(self, frame):
        # The preview already buffers live pixels. Bridge one ambiguous sample
        # only if both nearby detections agree on the same region in this shot.
        older = newer = None
        for index, pts, shot, snapshot in self._results:
            if shot != frame.shot or abs(pts - frame.pts) > 0.13:
                continue
            boxes = snapshot.active(pts)
            if not boxes:
                continue
            if index < frame.index:
                older = pts, boxes
            elif index > frame.index:
                newer = pts, boxes
                break
        if older is None or newer is None or not 0 < newer[0] - older[0] <= 0.15:
            return ()
        fraction = (frame.pts - older[0]) / (newer[0] - older[0])
        interpolated = []
        for box in older[1]:
            match = max(newer[1], key=lambda candidate: _iou(box, candidate))
            if _iou(box, match) >= 0.32:
                interpolated.append(tuple(round(a + fraction * (b - a)) for a, b in zip(box, match)))
        return interpolated

    def _run(self):
        detector = None
        tracker = MaskTracker(retain_unmatched=False)
        try:
            detector = self._factory()
            if not getattr(detector, "available", True):
                raise RuntimeError(getattr(detector, "error", "Model unavailable"))
            print(f"Detector ready: {self.name}", flush=True)
        except Exception as exc:
            self.failed = True
            print(f"Detector unavailable ({self.name}): {exc}", file=sys.stderr, flush=True)
        finally:
            self.ready.set()
        if detector is None or self.failed:
            return

        previous_shot = None
        advance = getattr(detector, "advance", None)
        previous_index = None
        try:
            while not self.stopped.is_set():
                with self._changed:
                    self._changed.wait_for(
                        lambda: self._jobs or self.stopped.is_set(), timeout=0.15
                    )
                    if self.stopped.is_set():
                        return
                    if not self._jobs:
                        continue
                    frame, due = self._jobs.popleft()
                discontinuous = self.every_frame and previous_index is not None and frame.index != previous_index + 1
                previous_index = frame.index
                if frame.shot != previous_shot or discontinuous:
                    tracker.clear()
                    reset = getattr(detector, "reset", None)
                    if reset is not None:
                        reset()
                    previous_shot = frame.shot
                    due = True
                if due:
                    boxes = detector.detect(frame.pixels)
                elif advance is not None:
                    boxes = advance(frame.pixels)
                else:
                    continue
                tracker.update(boxes, frame.pts)
                snapshot = tracker.snapshot()
                with self._changed:
                    self._results.append((frame.index, frame.pts, frame.shot, snapshot))
                    self.samples += int(due)
                    self.tracking_updates += int(not due)
        except Exception as exc:
            self.failed = True
            with self._changed:
                self._jobs.clear()
                self._results.clear()
            print(f"Detector stopped ({self.name}): {exc}", file=sys.stderr, flush=True)


def police_detector():
    from police_vehicles import PoliceVehicleDetector
    return PoliceVehicleDetector()


def blood_detector():
    from blood_masking import BloodMosaicDetector
    return BloodMosaicDetector()


class SceneAdvisory:
    """Optional scene notices for the operator's console."""

    def __init__(self):
        from scene_guard import SceneGuard
        self.guard = SceneGuard()
        self.available = self.guard.available
        self.error = self.guard.error
        self.reset()

    def reset(self):
        self.previous = ""
        self.repeats = 0

    def detect(self, frame):
        decision = self.guard.analyze(frame)
        if not decision.available:
            raise RuntimeError(self.guard.error)
        label = decision.category if decision.margin >= 0.035 else ""
        if label != self.previous:
            self.previous = label
            self.repeats = 0
        self.repeats += 1
        if label and self.repeats == 2:
            print(f"Scene advisory: {label}. Press F8 if needed.", flush=True)
        return ()


class PreviewPipeline:
    """Decode and filter ahead at bounded memory cost; never wait for detection."""

    def __init__(self, args, source_factory=None):
        if source_factory is None:
            from video_presenter import open_preview_source
            source_factory = open_preview_source
        self.args = args
        self.size = args.width, args.height
        self.stopped = threading.Event()
        self.capture_started = threading.Event()
        self.input_ready = threading.Event()
        self.ended = threading.Event()
        self.fps = 30.0
        self.preload = 1
        self.live_delay = args.delay
        self.decoded = 0
        self.error = ""
        self.backend = ""
        self.frames = None
        self.audio = None
        if sys.platform == "win32" and args.file is not None and not getattr(args, "mute", False):
            from local_audio import LocalAudio
            self.audio = LocalAudio(args.file)
        self.police = MaskChannel("police vehicles", police_detector, 12, args.delay, self.stopped, every_frame=True)
        self.regions = MaskChannel("exposed regions", LocalMaskDetector, 18, args.delay, self.stopped, bridge_gap=True)
        self.blood = (
            MaskChannel("red blood regions", blood_detector, 12, args.delay, self.stopped)
            if args.blood_mode != "off" else None
        )
        self.advisory = (
            MaskChannel("scene advisory", SceneAdvisory, 3, args.delay, self.stopped)
            if args.scene_mode == "full" else None
        )
        self.channels = tuple(
            channel for channel in (self.police, self.regions, self.blood, self.advisory)
            if channel is not None
        )
        self._worker = threading.Thread(
            target=self._decode, args=(source_factory,), name="decode-video", daemon=True
        )
        self._worker.start()

    def _decode(self, source_factory):
        source = None
        try:
            source = source_factory(self.args)
            self.fps = fps_of(source)
            expected = source.get(cv2.CAP_PROP_FRAME_COUNT)
            self.police.set_input_fps(self.fps)
            self.backend = getattr(source, "backend", type(source).__name__)
            # Fit in 256 MiB while keeping the two-second 720p default intact.
            max_frames = max(2, (256 * 1024 * 1024) // (self.size[0] * self.size[1] * 3))
            wanted = max(1, math.ceil(self.args.delay * self.fps))
            capacity = min(max_frames, max(wanted + 6, 8))
            self.preload = min(wanted, max(1, capacity - 2))
            self.live_delay = min(self.args.delay, self.preload / self.fps)
            self.frames = queue.Queue(maxsize=capacity)
            color = ColorFilter(self.args.blood_mode)
            print(
                f"Input ready: {self.fps:.6g} fps, {self.backend}. "
                "Press F9 or G to start.", flush=True
            )
            if wanted > self.preload:
                print(f"Buffer shortened to {self.preload / self.fps:.2f}s at this size.", flush=True)
            self.input_ready.set()

            if self.args.file is None:
                while not self.stopped.is_set() and not self.capture_started.wait(0.1):
                    pass

            previous_pts = None
            boundaries = ShotBoundaries()
            shot = 0
            while not self.stopped.is_set():
                ok, image = source.read()
                captured = time.perf_counter()
                if not ok:
                    if self.args.file is None or not self.decoded:
                        raise RuntimeError("Input did not deliver a video frame.")
                    if math.isfinite(expected) and self.decoded + 1 < expected:
                        raise RuntimeError(f"Input ended at {self.decoded} of {expected:g} expected frames.")
                    break

                index = self.decoded
                if self.args.file is not None:
                    pts = source.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if not math.isfinite(pts) or pts < 0:
                        pts = 0 if previous_pts is None else previous_pts + 1 / self.fps
                    if previous_pts is not None and pts <= previous_pts:
                        pts = previous_pts + 1 / self.fps
                else:
                    pts = index / self.fps
                previous_pts = pts

                image = fit(image, self.size)
                if boundaries.advance(image):
                    shot += 1
                frame = VideoFrame(index, pts, shot, captured, image)
                for channel in self.channels:
                    channel.submit(frame)
                filtered, changed = color.apply(image)
                if changed:
                    frame = VideoFrame(index, pts, shot, captured, filtered, True)
                while not self.stopped.is_set():
                    try:
                        self.frames.put(frame, timeout=0.08)
                        self.decoded += 1
                        break
                    except queue.Full:
                        pass
        except Exception as exc:
            self.error = str(exc)
        finally:
            if source is not None:
                try:
                    source.release()
                except Exception as exc:
                    if not self.error and not self.stopped.is_set():
                        self.error = str(exc)
            self.input_ready.set()
            self.ended.set()

    def composite(self, frame):
        police = self.police.boxes_for(frame)
        exposed = self.regions.boxes_for(frame)
        blood = self.blood.boxes_for(frame) if self.blood is not None else ()
        exposed, blood, police = bounded_boxes(frame.pixels.shape, exposed, blood, police)
        if not police and not exposed and not blood:
            return frame.pixels, 0, 0, 0
        output = frame.pixels.copy()
        mosaic(output, police, blocks=6)
        mosaic(output, blood, solid=self.args.cover_style == "solid", blocks=4)
        mosaic(output, exposed, solid=self.args.cover_style == "solid")
        return output, len(police), len(exposed), len(blood)

    def describe(self):
        return {
            "decoded": self.decoded,
            "source_fps": self.fps,
            "backend": self.backend,
            "detectors": {
                c.name: {"samples": c.samples, "tracking_frames": c.tracking_updates,
                         "missed_work_items": c.dropped_work, "failed": c.failed}
                for c in self.channels
            },
        }

    def close(self):
        self.stopped.set()
        self.capture_started.set()
        if self.audio is not None:
            self.audio.close()


class PreviewWindow:
    def __init__(self, root, args, pipeline, presenter, stop_at_end=False):
        self.root = root
        self.args = args
        self.pipeline = pipeline
        self.presenter = presenter
        self.audio = getattr(pipeline, "audio", None)
        self.stop_at_end = stop_at_end
        self.pending = None
        self.requested = args.start
        self.started = False
        self.manual_cover = False
        self.cover_captures_before = -math.inf
        self.last_output = None
        self.origin = None
        self.first_pts = None
        self.last_visible_pts = None
        self.first_frame_time = None
        self.last_frame_time = None
        self.first_index = None
        self.last_index = None
        self.displayed = 0
        self.sequence_gaps = 0
        self.police_frames = 0
        self.region_frames = 0
        self.blood_frames = 0
        self.de_red_frames = 0
        self.held_frames = 0
        self.max_late = 0.0
        self.max_update_gap = 0.0
        self.ended = False
        self.closed = False
        self.ready_announced = False
        self._end_playback_scheduling = None
        self.events = queue.SimpleQueue()
        self.hotkeys = None

        self.standby = card((args.width, args.height))
        presenter.show(self.standby)
        root.protocol("WM_DELETE_WINDOW", self.close)
        for key in ("b", "B", "space", "F8"):
            root.bind(f"<KeyPress-{key}>", self.hold)
        for key in ("g", "G", "F9"):
            root.bind(f"<KeyPress-{key}>", self.resume)
        for key in ("q", "Q", "Escape"):
            root.bind(f"<KeyPress-{key}>", self.close)
        if sys.platform == "win32" and not args.local_keys:
            self.hotkeys = WindowsKeys(self.events)
            if not self.hotkeys.ok:
                print("Global F8/F9 unavailable. Focus PROGRAM and press B/G.", flush=True)
        root.after(5, self.tick)

    def hold(self, *_):
        if self.ended or self.manual_cover:
            return "break"
        self.manual_cover = True
        self.requested = True
        if self.last_output is not None:
            self.presenter.show(manual_mosaic(self.last_output))
        self.pipeline.capture_started.set()
        print("Program: MANUAL MOSAIC", flush=True)
        return "break"

    def resume(self, *_):
        if self.ended:
            return "break"
        if self.started and self.manual_cover:
            if self.args.file is None:
                # Keep the buffered captures filtered until footage taken after F9.
                self.cover_captures_before = time.perf_counter()
            elif self.last_output is not None:
                self.presenter.show(self.last_output)
        self.manual_cover = False
        self.requested = True
        self.pipeline.capture_started.set()
        return "break"

    def _release_playback_scheduling(self):
        if self._end_playback_scheduling is not None:
            self._end_playback_scheduling()
            self._end_playback_scheduling = None

    def close(self, *_):
        if not self.closed:
            self.closed = True
            self._release_playback_scheduling()
            self.pipeline.close()
            if self.hotkeys is not None:
                self.hotkeys.close()
            self.root.destroy()
        return "break"

    def report(self):
        span = (
            self.last_frame_time - self.first_frame_time
            if self.first_frame_time is not None and self.last_frame_time is not None else 0.0
        )
        return dict(
            self.pipeline.describe(),
            displayed=self.displayed,
            delivered_fps=(self.displayed - 1) / span if span > 0 else 0.0,
            source_first_index=self.first_index,
            source_last_index=self.last_index,
            source_first_pts=self.first_pts,
            source_last_pts=self.last_visible_pts,
            sequence_gaps=self.sequence_gaps,
            police_mosaic_frames=self.police_frames,
            exposure_mosaic_frames=self.region_frames,
            blood_mosaic_frames=self.blood_frames,
            selective_color_frames=self.de_red_frames,
            automatic_full_cover_frames=0,
            manual_cover_frames=self.held_frames,
            manual_mosaic_frames=self.held_frames,
            maximum_late_ms=round(self.max_late * 1000, 2),
            maximum_update_gap_ms=round(self.max_update_gap * 1000, 2),
            input_completed=self.ended and not self.pipeline.error,
            error=self.pipeline.error,
        )

    def tick(self):
        while True:
            try:
                event, _ = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "hold":
                self.hold()
            elif event == "resume":
                self.resume()

        if self.ended:
            self.root.after(25, self.tick)
            return
        pipeline = self.pipeline
        if pipeline.frames is None:
            if pipeline.ended.is_set():
                self.finish()
            else:
                self.root.after(10, self.tick)
            return
        if not self.ready_announced and all(c.ready.is_set() for c in pipeline.channels):
            if self.audio is None or self.audio.ready.is_set():
                self.ready_announced = True
                if self.audio is not None:
                    if self.audio.state == "ready":
                        print("Local original audio: ready on the Windows playback device.", flush=True)
                    elif self.audio.error:
                        print(f"Local original audio unavailable: {self.audio.error}", file=sys.stderr, flush=True)
                    else:
                        print("Local file has no audio track.", flush=True)
                print("Program: ready. F9 starts normal playback; F8 starts the manual mosaic.", flush=True)
        if not self.requested:
            self.root.after(20, self.tick)
            return
        if not self.started:
            pipeline.capture_started.set()
            enough_frames = pipeline.frames.qsize() >= pipeline.preload or pipeline.ended.is_set()
            first_masks_ready = all(
                c.failed or c.samples > 0 or not pipeline.decoded for c in pipeline.channels
            )
            if not (enough_frames and self.ready_announced and first_masks_ready):
                self.root.after(5, self.tick)
                return
            from video_presenter import start_playback_scheduling
            self._end_playback_scheduling = start_playback_scheduling()
            self.started = True
            print("Program: playing at input cadence.", flush=True)

        if self.pending is None:
            try:
                self.pending = pipeline.frames.get_nowait()
            except queue.Empty:
                if pipeline.ended.is_set():
                    self._finish_after_last_frame()
                else:
                    self.root.after(2, self.tick)
                return

        frame = self.pending
        now = time.perf_counter()
        first_frame = self.origin is None
        if first_frame:
            self.origin = now
            self.first_pts = frame.pts
        if self.args.file is not None:
            elapsed = frame.pts - self.first_pts
            audio_position = self.audio.position() if self.audio is not None else None
            deadline = self.origin + elapsed if audio_position is None else now + elapsed - audio_position
        else:
            deadline = frame.captured + pipeline.live_delay
        present_after = deadline
        if self.args.file is not None and self.last_frame_time is not None:
            # Avoid bunching several frames into one desktop refresh after a stall.
            present_after = max(present_after, self.last_frame_time + 0.5 / pipeline.fps)
        wait = present_after - now
        if wait > 0.001:
            self.root.after(max(1, min(20, int(wait * 1000))), self.tick)
            return

        output, police, exposed, blood = pipeline.composite(frame)
        self.last_output = output
        if self.manual_cover or frame.captured < self.cover_captures_before:
            self.held_frames += 1
            output = manual_mosaic(output)
        if first_frame and self.audio is not None:
            self.origin = time.perf_counter()
            self.audio.start()
        self.presenter.show(output)
        self.police_frames += bool(police)
        self.region_frames += bool(exposed)
        self.blood_frames += bool(blood)
        self.de_red_frames += frame.color_changed
        shown = time.perf_counter()
        self.max_late = max(self.max_late, max(0.0, shown - deadline))
        if self.last_frame_time is not None:
            self.max_update_gap = max(self.max_update_gap, shown - self.last_frame_time)
        if self.last_index is not None and frame.index != self.last_index + 1:
            self.sequence_gaps += max(1, frame.index - self.last_index - 1)
        if self.first_index is None:
            self.first_index = frame.index
            self.first_frame_time = shown
        self.last_index = frame.index
        self.last_visible_pts = frame.pts
        self.last_frame_time = shown
        self.displayed += 1
        self.pending = None
        self.root.after(1, self.tick)

    def _finish_after_last_frame(self):
        if self.last_frame_time is not None:
            remaining = self.last_frame_time + 1 / self.pipeline.fps - time.perf_counter()
            if remaining > 0:
                self.root.after(max(1, math.ceil(remaining * 1000)), self._finish_after_last_frame)
                return
        if not self.closed:
            self.finish()

    def finish(self):
        self.ended = True
        self._release_playback_scheduling()
        if self.audio is not None:
            self.audio.close()
        if self.pipeline.error:
            print(f"Input error: {self.pipeline.error}", file=sys.stderr, flush=True)
        else:
            print("Program: end of file. Reopen the file to replay.", flush=True)
        print_summary(self.report())
        if self.stop_at_end:
            self.close()
        else:
            self.presenter.show(self.standby)
            self.root.after(25, self.tick)


def print_summary(summary):
    import json
    print(f"Playback summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)


def run_preview(args, *, source_factory=None, presenter_factory=None, stop_at_end=False):
    import tkinter as tk

    if presenter_factory is None:
        from video_presenter import Presenter
        presenter_factory = Presenter
    before = cv2.getNumThreads()
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(min(0.002, switch_interval))
    cv2.setNumThreads(min(2, before))
    windows_timer = None
    if sys.platform == "win32":
        import ctypes
        timer = ctypes.WinDLL("winmm")
        if timer.timeBeginPeriod(1) == 0:
            windows_timer = timer
    root = None
    pipeline = None
    window = None
    try:
        root = tk.Tk()
        root.title(WINDOW)
        root.resizable(False, False)
        root.configure(bg="black")
        presenter = presenter_factory(root, args.width, args.height)
        pipeline = PreviewPipeline(args, source_factory)
        window = PreviewWindow(root, args, pipeline, presenter, stop_at_end)
        root.mainloop()
        result = window.report()
        if not window.ended:
            print_summary(result)
        return result
    finally:
        if window is not None and not window.closed:
            window.close()
        elif pipeline is not None:
            pipeline.close()
        if pipeline is not None and pipeline.audio is not None:
            pipeline.audio.close(wait=True)
        cv2.setNumThreads(before)
        sys.setswitchinterval(switch_interval)
        if windows_timer is not None:
            windows_timer.timeEndPeriod(1)
