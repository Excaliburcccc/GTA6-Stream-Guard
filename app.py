"""Local, video-only program window for OBS, or frame-by-frame MP4 export."""

import argparse
from collections import deque
from functools import lru_cache
import math
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

from detector import FrameGuard


WINDOW = "GTA6 Stream Guard - PROGRAM"


@lru_cache(maxsize=1)
def standby_image():
    path = Path(__file__).resolve().with_name("standby.png")
    try:
        # imdecode supports Windows folders containing non-ASCII characters.
        data = np.frombuffer(path.read_bytes(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


@lru_cache(maxsize=4)
def card(size):
    image = standby_image()
    if image is not None:
        frame = fit(image, size)
    else:
        w, h = size
        frame = np.empty((h, w, 3), dtype=np.uint8)
        frame[:] = (32, 27, 23)
        cv2.putText(frame, "ON HOLD", (w // 14, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, w / 1000), (225, 226, 227), max(1, w // 550), cv2.LINE_AA)
    frame.setflags(write=False)
    return frame


def fit(frame, size):
    w, h = size
    fh, fw = frame.shape[:2]
    if (fw, fh) == size:
        return frame
    ratio = min(w / fw, h / fh)
    nw, nh = max(1, round(fw * ratio)), max(1, round(fh * ratio))
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    image = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    image[y:y + nh, x:x + nw] = small
    return image


def checked_frame(guard, frame):
    image, action = guard.process(frame)
    if (not isinstance(image, np.ndarray) or image.shape != frame.shape
            or image.dtype != np.uint8 or not isinstance(action, str) or not action):
        raise RuntimeError("Invalid detector result")
    if action.startswith("hold:"):
        h, w = frame.shape[:2]
        return card((w, h)), action
    return image, action


def open_source(args):
    if args.file is not None:
        source = cv2.VideoCapture(str(args.file))
    elif sys.platform == "win32":
        source = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not source.isOpened():
            source.release()
            source = cv2.VideoCapture(args.camera)
    else:
        source = cv2.VideoCapture(args.camera)
    if not source.isOpened():
        source.release()
        raise RuntimeError("Could not open input. Check the file/device number and close other capture apps.")
    if args.file is None:
        source.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        source.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return source


def get_guard(args):
    guard = FrameGuard(blood_mode=args.blood_mode, scene_mode=args.scene_mode, cover_style=args.cover_style)
    if getattr(guard, "startup_error", None):
        raise RuntimeError(str(guard.startup_error))
    return guard


def fps_of(source):
    fps = source.get(cv2.CAP_PROP_FPS)
    return fps if math.isfinite(fps) and 0 < fps <= 240 else 30.0


REVIEW_MEMORY = 160 * 1024 * 1024
LOOKBACK = 0.6


class ReviewBuffer:
    """Bounded processed-frame storage; source timestamps are monotonic."""
    def __init__(self, size, delay, live):
        self.size = size
        self.delay = delay
        self.lookback = min(LOOKBACK, delay)
        self.replacement = card(size)
        self.live = live
        self.capacity = max(1, min(180, REVIEW_MEMORY // self.replacement.nbytes))
        self.lock = threading.Lock()
        self.epoch = -1
        self.accept_after = -math.inf
        self.pending = deque()
        self.clear(-1)

    def clear(self, epoch, accept_after=-math.inf):
        with self.lock:
            self.accept_after = accept_after
            self.epoch = epoch
            self.pending.clear()
            self.stored = 0
            self.latest = -math.inf
            self.last_saved = -math.inf
            self.was_covered = False

    def submit(self, epoch, timestamp, image, action):
        with self.lock:
            if self.epoch != epoch or timestamp < self.accept_after:
                return
            self.latest = timestamp
            covered = action.startswith("cover")
            sensitive = action.startswith("hold") or (covered and not self.was_covered)
            self.was_covered = covered
            if sensitive and self.lookback:
                for entry in reversed(self.pending):
                    if entry[0] + 1e-8 < timestamp - self.lookback:
                        break
                    if entry[1] is not None:
                        entry[1:] = [None, "hold:lookback"]
                        self.stored -= 1
            # An operator's delay always wins over displaying more frames.
            if self.live and (timestamp - self.last_saved < self.delay / self.capacity
                              or len(self.pending) >= self.capacity):
                return
            if self.stored < self.capacity:
                self.stored += 1
            else:
                # Offline high-resolution/high-fps sources retain timing by using
                # the shared safe card if the optional lookback exceeds the budget.
                image, action = None, "hold:memory-limit"
            self.pending.append([timestamp, image, action])
            self.last_saved = timestamp

    def pop_ready(self, now, ended=False):
        with self.lock:
            if not self.pending:
                return None
            timestamp, frame, action = self.pending[0]
            if timestamp + self.delay > now:
                return None
            if not ended and timestamp + self.lookback > self.latest + 1e-8:
                return None
            self.pending.popleft()
            if frame is None:
                frame = self.replacement
            else:
                self.stored -= 1
            return frame, action

    def status(self):
        with self.lock:
            return self.latest, bool(self.pending)


def export_video(args):
    if args.export.exists():
        raise ValueError("Output already exists. Choose a new MP4 filename.")
    guard = get_guard(args)
    source = open_source(args)
    writer = None
    temporary = None
    counts = {}
    frames = 0
    try:
        fps = fps_of(source)
        reported = source.get(cv2.CAP_PROP_FRAME_COUNT)
        while True:
            ok, raw = source.read()
            if not ok:
                break
            frame, action = checked_frame(guard, raw)
            if writer is None:
                with tempfile.NamedTemporaryFile(dir=args.export.parent, suffix=".mp4", delete=False) as tmp:
                    temporary = Path(tmp.name)
                # Software encoders typically require even dimensions.
                h, w = frame.shape[:2]
                size = ((w + 1) // 2 * 2, (h + 1) // 2 * 2)
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError("Could not initialize MP4 encoder")
                review = ReviewBuffer(size, min(args.delay, LOOKBACK), live=False)
            timestamp = frames / fps
            review.submit(-1, timestamp, fit(frame, size), action)
            frames += 1
            while (ready_frame := review.pop_ready(timestamp)) is not None:
                image, outgoing = ready_frame
                writer.write(image)
                counts[outgoing] = counts.get(outgoing, 0) + 1
        if not frames or (math.isfinite(reported) and frames + 0.5 < reported):
            raise RuntimeError(f"Incomplete input. Decoded {frames} frames (container reports {reported}).")
        while (ready_frame := review.pop_ready(math.inf, ended=True)) is not None:
            image, outgoing = ready_frame
            writer.write(image)
            counts[outgoing] = counts.get(outgoing, 0) + 1
        writer.release()
        writer = None
        if not temporary.stat().st_size:
            raise RuntimeError("Encoder produced no data")
        # An exclusive destination avoids replacing an existing file.
        if sys.platform == "win32":
            os.rename(temporary, args.export)
        else:
            os.link(temporary, args.export)
        print(f"Exported {frames} silent frames to {args.export}. Actions: {counts}")
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Switch:
    """Invalidate any in-flight result from before a hold or resume."""
    def __init__(self):
        self.lock = threading.Lock()
        self.live = False
        self.epoch = 0
        self.ever_started = threading.Event()

    def set(self, live):
        with self.lock:
            self.live = live
            self.epoch += 1
            if live:
                self.ever_started.set()

    def snapshot(self):
        with self.lock:
            return self.live, self.epoch


class CameraFeed:
    """Drain the capture device at its cadence; retain only its newest raw frame."""
    def __init__(self, source, stopped, events):
        self.source = source
        self.stopped = stopped
        self.events = events
        self.condition = threading.Condition()
        self.latest = None
        self.failed = False
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def read_loop(self):
        try:
            while not self.stopped.is_set() and not self.closing.is_set():
                ok, image = self.source.read()
                if not ok:
                    raise RuntimeError("Capture device disconnected or stopped delivering frames.")
                with self.condition:
                    self.latest = (time.monotonic(), image)
                    self.condition.notify()
        except Exception as exc:
            with self.condition:
                self.failed = True
                self.latest = None
                self.condition.notify()
            self.events.put(("fault", str(exc)))
        finally:
            self.source.release()

    def take(self):
        with self.condition:
            while self.latest is None and not self.failed and not self.stopped.is_set():
                self.condition.wait(0.15)
            packet = self.latest
            self.latest = None
            return packet


def capture_loop(args, switch, stopped, events, review):
    source = None
    feed = None
    try:
        guard = get_guard(args)
        source = open_source(args)
        period = 1 / fps_of(source)
        events.put(("ready", "Input and detector ready. Press F9 or focus PROGRAM and press G."))
        if args.file is None:
            feed = CameraFeed(source, stopped, events)
        while not stopped.is_set() and not switch.ever_started.wait(0.1):
            if feed and feed.failed:
                return
        decoded = 0
        last_model_hold = None
        reported = source.get(cv2.CAP_PROP_FRAME_COUNT) if args.file else 0
        while not stopped.is_set():
            live, epoch = switch.snapshot()
            if feed is not None:
                packet = feed.take()
                if packet is None:
                    return
                captured_at, raw = packet
            else:
                captured_at = time.monotonic()
                ok, raw = source.read()
                if not ok:
                    complete = decoded and (reported <= 0 or decoded + 0.5 >= reported)
                    events.put(("eof" if complete else "fault", "Input ended."))
                    return
                decoded += 1
            if live:
                checked, action = checked_frame(guard, raw)
                model_hold = action if action.startswith("hold:") else None
                if model_hold != last_model_hold:
                    print(f"Detector: {model_hold or 'hold cleared'}", flush=True)
                    last_model_hold = model_hold
                review.submit(epoch, captured_at, fit(checked, (args.width, args.height)), action)
            if feed is None:
                stopped.wait(max(0, period - (time.monotonic() - captured_at)))
    except Exception as error:
        events.put(("fault", str(error)))
    finally:
        if feed is not None:
            feed.closing.set()
        elif source is not None:
            source.release()


class WindowsKeys:
    """Use a dedicated Win32 message queue; Tk cannot consume these hotkeys."""
    def __init__(self, events):
        self.events = events
        self.ready = threading.Event()
        self.ok = False
        self.thread_id = None
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()
        self.ready.wait(3)

    def listen(self):
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        ids = []
        try:
            for ident, vk in [(1, 0x77), (2, 0x78)]:  # F8 and F9
                if not u32.RegisterHotKey(None, ident, 0x4000, vk):  # MOD_NOREPEAT
                    return
                ids.append(ident)
            self.ok = True
            self.ready.set()
            msg = wintypes.MSG()
            while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == 0x0312:
                    self.events.put(("resume" if msg.wParam == 2 else "hold", ""))
        finally:
            for ident in ids:
                u32.UnregisterHotKey(None, ident)
            self.ready.set()

    def close(self):
        import ctypes
        if self.thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
            self.thread.join(timeout=0.5)


def preview(args):
    import tkinter as tk
    root = tk.Tk()
    root.title(WINDOW)
    root.resizable(False, False)
    root.configure(bg="black")
    size = args.width, args.height
    label = tk.Label(root, borderwidth=0, highlightthickness=0, bg="black")
    label.pack()
    events = queue.SimpleQueue()
    review = ReviewBuffer(size, args.delay, live=True)
    ended = False
    switch = Switch()
    stopped = threading.Event()
    hotkeys = None
    ready = False
    last_action = None
    deadline = 0.0

    def show(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        photo = tk.PhotoImage(data=f"P6\n{w} {h}\n255\n".encode() + rgb.tobytes(), format="PPM")
        label.configure(image=photo)
        label.image = photo

    def hold(message="ON HOLD"):
        nonlocal last_action
        switch.set(False)
        review.clear(switch.snapshot()[1])
        last_action = None
        print(f"Program: {message}", flush=True)
        show(card(size))

    def resume():
        nonlocal deadline
        if ready and not switch.snapshot()[0]:
            switch.set(True)
            review.clear(switch.snapshot()[1], time.monotonic())
            deadline = time.monotonic() + args.timeout
            print("Program: RESUMING - waiting for reviewed frames", flush=True)
            # Leave the safe card visible until a verified result arrives.

    def tick():
        nonlocal ready, last_action, deadline, ended
        while True:
            try:
                event, detail = events.get_nowait()
            except queue.Empty:
                break
            if detail:
                print(detail, flush=True)
            if event == "ready":
                ready = True
                hold("READY - ON HOLD")
                if args.start:
                    resume()
            elif event == "eof":
                ended = True
                ready = False
            elif event == "fault":
                ready = False
                ended = False
                hold("INPUT UNAVAILABLE")
            elif event == "hold":
                hold()
            elif event == "resume":
                resume()
        if switch.snapshot()[0]:
            now = time.monotonic()
            latest, pending = review.status()
            deadline = max(deadline, latest + args.timeout)
            if ended and not pending:
                hold("INPUT ENDED")
            elif not ended and now > deadline:
                hold("TIMEOUT - ON HOLD")
                print("No fresh processed frame within timeout. Press F9/G to retry.", flush=True)
            else:
                newest = None
                while (item := review.pop_ready(now, ended)) is not None:
                    newest = item
                if newest is not None:
                    frame, action = newest
                    show(frame)
                    if action != last_action:
                        print(f"Action: {action}", flush=True)
                        last_action = action
        root.after(10, tick)

    def close(*_):
        stopped.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    for key in ("b", "B", "space", "F8"):
        root.bind(f"<KeyPress-{key}>", lambda _: hold())
    for key in ("g", "G", "F9"):
        root.bind(f"<KeyPress-{key}>", lambda _: resume())
    for key in ("q", "Q", "Escape"):
        root.bind(f"<KeyPress-{key}>", close)
    try:
        print("Program: LOADING - ON HOLD", flush=True)
        show(card(size))
        root.update_idletasks()
        if sys.platform == "win32" and not args.local_keys:
            hotkeys = WindowsKeys(events)
            if not hotkeys.ok:
                raise RuntimeError("F8/F9 global hotkeys unavailable. Free these keys or use --local-keys.")
        worker = threading.Thread(target=capture_loop, args=(args, switch, stopped, events, review), daemon=True)
        worker.start()
        root.after(10, tick)
        root.mainloop()
    finally:
        stopped.set()
        if hotkeys:
            hotkeys.close()
        try:
            root.destroy()
        except tk.TclError:
            pass


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--camera", type=int, default=0, help="capture card/webcam index (default 0)")
    sources.add_argument("--file", type=Path, help="decode local video instead of the capture card")
    parser.add_argument("--width", type=int, default=1280, help="program window width")
    parser.add_argument("--height", type=int, default=720, help="program window height")
    parser.add_argument("--blood-mode", choices=("always", "selective", "off"), default="selective",
                        help="selective red filtering, grayscale, or no color filter")
    parser.add_argument("--scene-mode", choices=("full", "basic"), default=None,
                        help="full enables console scene notices in preview; basic uses local detectors only")
    parser.add_argument("--cover-style", choices=("mosaic", "solid"), default="mosaic",
                        help="appearance of detected exposure regions")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="editable pre-output buffer in seconds, 0..10 (default 2)")
    parser.add_argument("--timeout", type=float, default=3.0, help="legacy seconds without fresh results before manual hold")
    parser.add_argument("--local-keys", action="store_true", help="only accept keys while PROGRAM has focus")
    parser.add_argument("--legacy", action="store_true", help="use the previous per-frame scene blocking pipeline")
    parser.add_argument("--software-decode", action="store_true", help="use OpenCV for local video decoding")
    parser.add_argument("--mute", action="store_true", help="mute original audio during local-file preview")
    parser.add_argument("--start", action="store_true", help="automatically begin when input and detector are ready")
    parser.add_argument("--headless", action="store_true", help="offline export; requires --file --export --start")
    parser.add_argument("--export", type=Path, help="new silent MP4 for offline inspection")
    args = parser.parse_args()
    if args.scene_mode is None:
        args.scene_mode = "full" if args.legacy else "basic"
    if args.legacy and args.blood_mode == "off":
        parser.error("--legacy supports selective or grayscale only")
    if not (64 <= args.width <= 4096 and 64 <= args.height <= 4096):
        parser.error("program dimensions must be 64..4096")
    if not math.isfinite(args.delay) or not 0 <= args.delay <= 10:
        parser.error("--delay must be finite and in 0..10 seconds")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    if args.file and not args.file.is_file():
        parser.error("input file does not exist")
    if args.headless or args.export:
        if not (args.headless and args.start and args.file and args.export):
            parser.error("offline export requires --headless --start --file INPUT --export OUTPUT")
        if args.export.suffix.lower() != ".mp4":
            parser.error("output must have .mp4 extension")
    return args


def main():
    args = arguments()
    try:
        if args.legacy:
            if args.headless:
                export_video(args)
            else:
                preview(args)
        elif args.headless:
            from local_export import export_local_video
            export_local_video(args)
        else:
            from smooth_preview import run_preview
            result = run_preview(args)
            if result["error"]:
                return 1
    except (RuntimeError, ValueError, OSError, ImportError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
