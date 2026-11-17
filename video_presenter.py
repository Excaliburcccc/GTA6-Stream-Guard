"""Video sources and the OBS-compatible PROGRAM surface for smooth preview."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import shutil
import subprocess
import sys
import threading

import cv2
import numpy as np
from PIL import Image, ImageTk


def _fit_dimensions(source_width, source_height, width, height):
    scale = min(width / source_width, height / source_height)
    return min(width, max(1, round(source_width * scale))), min(height, max(1, round(source_height * scale)))


class PillowPresenter:
    """Reuse one Tk image so each PROGRAM frame updates the same native window."""

    def __init__(self, root, width, height):
        import tkinter as tk

        self.width = width
        self.height = height
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._rgb = np.empty_like(self._canvas)
        self._photo = ImageTk.PhotoImage("RGB", size=(width, height), master=root)
        self.widget = tk.Label(root, image=self._photo, borderwidth=0, highlightthickness=0, bg="black")
        self.widget.pack()

    def show(self, frame):
        height, width = frame.shape[:2]
        if (width, height) != (self.width, self.height):
            scaled_width, scaled_height = _fit_dimensions(width, height, self.width, self.height)
            interpolation = cv2.INTER_AREA if scaled_width < width else cv2.INTER_LINEAR
            resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)
            x = (self.width - scaled_width) // 2
            y = (self.height - scaled_height) // 2
            self._canvas.fill(0)
            self._canvas[y:y + scaled_height, x:x + scaled_width] = resized
            frame = self._canvas
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb)
        self._photo.paste(Image.fromarray(self._rgb))
        self.widget.update_idletasks()


class Presenter:
    """Choose the native Windows surface, with Tk bitmap fallback."""

    def __init__(self, root, width, height):
        self._backend = None
        if sys.platform == "win32" and os.environ.get("GTA6_PREVIEW_RENDERER") != "tk":
            try:
                from win32_presenter import WindowsBitmapPresenter
                self._backend = WindowsBitmapPresenter(root, width, height)
                print("Video renderer: Windows bitmap", flush=True)
            except (ImportError, OSError) as exc:
                print(f"Video renderer fallback: {exc}", file=sys.stderr, flush=True)
        if self._backend is None:
            self._backend = PillowPresenter(root, width, height)
            print("Video renderer: Tk bitmap", flush=True)
        self.widget = self._backend.widget

    def show(self, frame):
        self._backend.show(frame)

def start_playback_scheduling():
    """Register only the calling playback thread, and release it on that thread."""
    if sys.platform != "win32":
        return None
    import ctypes as ct
    from ctypes import wintypes as wt

    try:
        avrt = ct.WinDLL("avrt")
        attach = avrt.AvSetMmThreadCharacteristicsW
        attach.argtypes = wt.LPCWSTR, ct.POINTER(wt.DWORD)
        attach.restype = wt.HANDLE
        release = avrt.AvRevertMmThreadCharacteristics
        release.argtypes = (wt.HANDLE,)
        release.restype = wt.BOOL
        task_index = wt.DWORD(0)
        task = attach("Playback", ct.byref(task_index))
    except (AttributeError, OSError):
        return None
    return (lambda: release(task)) if task else None

@dataclass(frozen=True)
class _VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    pixel_format: str
    sample_aspect: float


def _fraction(value, default=None):
    try:
        number = float(Fraction(str(value).replace(":", "/")))
        return number if math.isfinite(number) and number > 0 else default
    except (ValueError, ZeroDivisionError):
        return default


def _seconds(value):
    if not value:
        return None
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (AttributeError, TypeError, ValueError):
        return None


def _inspect(ffprobe, file):
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "V:0", "-show_streams", "-of", "json", str(file)],
        stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise ValueError("No video stream found")
    stream = streams[0]
    fps = _fraction(stream.get("avg_frame_rate")) or _fraction(stream.get("r_frame_rate"))
    if fps is None:
        raise ValueError("Source reports no nominal video frame rate")
    frames = stream.get("nb_frames")
    if frames and str(frames).isdigit():
        frame_count = int(frames)
    else:
        tags = stream.get("tags", {})
        duration = _fraction(stream.get("duration")) or _seconds(tags.get("DURATION"))
        frame_count = round(duration * fps) if duration else 0
    return _VideoInfo(
        int(stream["width"]), int(stream["height"]), fps, frame_count,
        stream.get("pix_fmt", ""), _fraction(stream.get("sample_aspect_ratio"), 1.0),
    )


# FFmpeg showinfo prints one presentation timestamp before each frame write.
_FRAME_PTS = re.compile(rb"\bn:\s*(\d+).*?\bpts_time:\s*([^\s]+)")
_CUDA_PIXEL_FORMATS = {
    "yuv420p": "nv12",
    "yuvj420p": "nv12",
    "nv12": "nv12",
    "yuv420p10le": "p010le",
    "p010le": "p010le",
}


class _CudaFileSource:
    """Read one BGR frame per decoded packet's presentation frame, with PTS."""

    backend = "ffmpeg-cuda"

    def __init__(self, ffmpeg, file, info, width, height):
        self.fps = info.fps
        self.frame_count = info.frames
        self.width, self.height = width, height
        self._index = 0
        self._first_frame = None
        self._first_pts = None
        self._position_ms = 0.0
        self._closed = threading.Event()
        self._pts = Queue(maxsize=32)
        self._errors = deque(maxlen=8)
        hw_format = _CUDA_PIXEL_FORMATS[info.pixel_format]
        filters = (
            f"scale_cuda={width}:{height}:interp_algo=bicubic:format={hw_format},"
            f"hwdownload,format={hw_format},format=bgr24,showinfo=checksum=0"
        )
        self._process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-nostats", "-xerror",
             "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(file),
             "-map", "0:V:0", "-an", "-sn", "-dn", "-filter_threads", "2",
             "-vf", filters, "-fps_mode", "passthrough", "-c:v", "rawvideo",
             "-threads:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1024 * 1024, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._reader = threading.Thread(target=self._read_timestamps, name="video-pts", daemon=True)
        self._reader.start()
        try:
            self._first_frame = self._read_frame()
            if self._first_frame is None:
                raise RuntimeError("CUDA decoder returned no video frames")
        except BaseException:
            self.release()
            raise

    def _read_timestamps(self):
        try:
            for line in self._process.stderr:
                if self._closed.is_set():
                    return
                match = _FRAME_PTS.search(line)
                if match is None:
                    if b"showinfo" not in line:
                        self._errors.append(line.decode("utf-8", errors="replace").strip())
                    continue
                try:
                    pts = float(match[2])
                except ValueError:
                    pts = None
                while not self._closed.is_set():
                    try:
                        self._pts.put((int(match[1]), pts), timeout=0.1)
                        break
                    except Full:
                        continue
        except (ValueError, OSError):
            pass
        except Exception as exc:
            self._errors.append(str(exc))

    def _read_frame(self):
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        size = self._process.stdout.readinto(frame)
        if size == frame.nbytes:
            return frame
        if self._closed.is_set():
            return None
        if size:
            raise RuntimeError(f"CUDA decoder returned a partial frame ({size}/{frame.nbytes} bytes)")
        code = self._process.wait(timeout=3)
        if code:
            self._reader.join(timeout=0.2)
            details = "\n".join(self._errors)
            raise RuntimeError(f"CUDA decoder exited with status {code}. {details}")
        return None

    def read(self):
        if self._closed.is_set():
            return False, None
        frame = self._first_frame
        self._first_frame = None
        if frame is None:
            frame = self._read_frame()
        if frame is None:
            return False, None

        pts = None
        try:
            while True:
                frame_number, candidate = self._pts.get(timeout=0.2)
                if frame_number == self._index:
                    pts = candidate
                    break
                if frame_number > self._index:
                    raise RuntimeError(f"CUDA timestamps skipped source frame {self._index}")
        except Empty:
            # Older FFmpeg builds may omit showinfo timestamps.
            pass
        if pts is not None and math.isfinite(pts):
            if self._first_pts is None:
                self._first_pts = pts - self._index / self.fps
            self._position_ms = 1000 * (pts - self._first_pts)
        else:
            self._position_ms = 1000 * self._index / self.fps
        self._index += 1
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self._index
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._position_ms
        return 0.0

    def isOpened(self):
        return not self._closed.is_set()

    def release(self):
        if self._closed.is_set():
            return
        self._closed.set()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process.stdout.close()
        self._process.stderr.close()
        self._reader.join(timeout=0.1)


class _OpenCVSource:
    def __init__(self, capture, backend, fallback_reason=None):
        self._capture = capture
        self.backend = backend
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count = round(reported_frames) if math.isfinite(reported_frames) and reported_frames > 0 else 0
        self.fallback_reason = fallback_reason

    def read(self):
        return self._capture.read()

    def get(self, prop):
        return self._capture.get(prop)

    def isOpened(self):
        return self._capture.isOpened()

    def release(self):
        self._capture.release()


def open_preview_source(args):
    """Open a source with a VideoCapture-style interface and backend diagnostics."""
    # Imported lazily so the preview module can be tested separately.
    from app import open_source

    if args.file is None:
        return _OpenCVSource(open_source(args), "opencv-camera")

    file = Path(args.file)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    why = None
    if ffmpeg and ffprobe and not getattr(args, "software_decode", False):
        try:
            info = _inspect(ffprobe, file)
            if info.pixel_format not in _CUDA_PIXEL_FORMATS:
                raise ValueError(f"CUDA preview does not handle {info.pixel_format}")
            width, height = _fit_dimensions(
                info.width * info.sample_aspect, info.height, args.width, args.height
            )
            # YUV 4:2:0 surfaces must be even in both axes.
            width = max(2, width - width % 2)
            height = max(2, height - height % 2)
            return _CudaFileSource(ffmpeg, file, info, width, height)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            why = str(exc).splitlines()[0]
    return _OpenCVSource(open_source(args), "opencv-file", why)
