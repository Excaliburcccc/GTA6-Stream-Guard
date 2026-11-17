"""Stream a local file's soundtrack to the Windows playback device."""

import ctypes as C
import json
import math
from queue import Empty, Full, Queue
import shutil
import subprocess
import sys
import threading
import time


_RATE = 48000
_CHANNELS = 2
_SAMPLE_BYTES = _CHANNELS * 2
_CHUNK_BYTES = (_RATE // 50) * _SAMPLE_BYTES  # 20 ms of stereo PCM.
_DEVICE_BUFFERS = 8
_POLL_INTERVAL = 0.004
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _media_audio(ffprobe, path):
    result = subprocess.run(
        [ffprobe, "-v", "error",
         "-show_entries", "stream=index,codec_type,start_time:stream_disposition=default,attached_pic:format=start_time",
         "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=8,
        check=True, creationflags=_CREATE_NO_WINDOW,
    )
    media = json.loads(result.stdout)
    streams = media.get("streams", ())
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        return None
    selected = next((s for s in audio if s.get("disposition", {}).get("default")), audio[0])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), {})
    for value in (video.get("start_time"), media.get("format", {}).get("start_time")):
        try:
            origin = float(value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(origin):
            return int(selected["index"]), origin
    return int(selected["index"]), 0.0


class _Decoder:
    def __init__(self, ffmpeg, path, stream, origin):
        self.blocks = Queue(maxsize=12)
        self.stopped = threading.Event()
        self.done = threading.Event()
        self.error = ""
        # Keep source timestamps, then start the PCM timeline at the first video
        # timestamp. The resampler pads delayed audio or trims early samples.
        filters = f"asetpts=PTS-({origin:.9f})/TB,aresample={_RATE}:async=1:first_pts=0"
        self.process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-xerror",
             "-copyts", "-i", str(path), "-map", f"0:{stream}", "-vn", "-sn", "-dn",
             "-filter_threads", "1", "-af", filters, "-ac", str(_CHANNELS),
             "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        self.reader = threading.Thread(target=self._read, name="audio-pcm", daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while not self.stopped.is_set():
                block = self.process.stdout.read(_CHUNK_BYTES)
                if not block:
                    code = self.process.wait(timeout=2)
                    if code:
                        raise RuntimeError(f"FFmpeg audio decoder exited with status {code}")
                    return
                if len(block) % _SAMPLE_BYTES:
                    raise RuntimeError("FFmpeg returned an incomplete audio sample")
                while not self.stopped.is_set():
                    try:
                        self.blocks.put(block, timeout=0.05)
                        break
                    except Full:
                        continue
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if not self.stopped.is_set():
                self.error = str(exc)
        finally:
            self.done.set()

    def close(self):
        self.stopped.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=0.5)
        self.reader.join(timeout=0.3)
        if not self.reader.is_alive():
            self.process.stdout.close()


class _WaveFormat(C.Structure):
    _pack_ = 1
    _fields_ = [
        ("tag", C.c_uint16), ("channels", C.c_uint16), ("samples_per_second", C.c_uint32),
        ("bytes_per_second", C.c_uint32), ("block_align", C.c_uint16),
        ("bits_per_sample", C.c_uint16), ("extra_size", C.c_uint16),
    ]


class _WaveHeader(C.Structure):
    _fields_ = [
        ("data", C.c_void_p), ("length", C.c_uint32), ("recorded", C.c_uint32),
        ("user", C.c_size_t), ("flags", C.c_uint32), ("loops", C.c_uint32),
        ("next", C.c_void_p), ("reserved", C.c_size_t),
    ]


class _WaveTime(C.Structure):
    # MMTIME's supported sample, byte and millisecond positions all use the
    # first DWORD of an eight-byte union.
    _fields_ = [("unit", C.c_uint32), ("value", C.c_uint32), ("unused", C.c_uint32)]


class _WaveBuffer:
    def __init__(self):
        self.storage = C.create_string_buffer(_CHUNK_BYTES)
        self.header = _WaveHeader(C.addressof(self.storage), _CHUNK_BYTES)
        self.busy = False


class _WaveOutput:
    """Own WinMM buffers and submit them from one worker thread."""

    _MAPPER = 0xFFFFFFFF
    _DONE = 1
    _TIME_MS = 1
    _TIME_SAMPLES = 2
    _TIME_BYTES = 4

    def __init__(self):
        self.api = C.WinDLL("winmm")
        signature = {
            "waveOutOpen": (C.POINTER(C.c_void_p), C.c_uint32, C.POINTER(_WaveFormat),
                            C.c_size_t, C.c_size_t, C.c_uint32),
            "waveOutGetPosition": (C.c_void_p, C.POINTER(_WaveTime), C.c_uint32),
        }
        for name in ("waveOutPause", "waveOutRestart", "waveOutReset", "waveOutClose"):
            signature[name] = (C.c_void_p,)
        for name in ("waveOutPrepareHeader", "waveOutWrite", "waveOutUnprepareHeader"):
            signature[name] = (C.c_void_p, C.POINTER(_WaveHeader), C.c_uint32)
        for name, arguments in signature.items():
            function = getattr(self.api, name)
            function.argtypes = arguments
            function.restype = C.c_uint32

        self.handle = C.c_void_p()
        self.buffers = []
        self.samples = 0
        audio_format = _WaveFormat(
            1, _CHANNELS, _RATE, _RATE * _SAMPLE_BYTES, _SAMPLE_BYTES, 16, 0
        )
        self._call("waveOutOpen", C.byref(self.handle), self._MAPPER, C.byref(audio_format), 0, 0, 0)
        try:
            self._call("waveOutPause", self.handle)
            for _ in range(_DEVICE_BUFFERS):
                buffer = _WaveBuffer()
                self._header("waveOutPrepareHeader", buffer)
                self.buffers.append(buffer)
        except BaseException:
            self.close()
            raise

    def _call(self, name, *arguments):
        result = getattr(self.api, name)(*arguments)
        if result:
            raise RuntimeError(f"{name} failed (Windows audio error {result})")

    def _header(self, name, buffer):
        self._call(name, self.handle, C.byref(buffer.header), C.sizeof(buffer.header))

    def free_buffers(self):
        available = []
        for buffer in self.buffers:
            if buffer.busy and buffer.header.flags & self._DONE:
                buffer.busy = False
            if not buffer.busy:
                available.append(buffer)
        return available

    def write(self, buffer, data):
        C.memmove(buffer.header.data, data, len(data))
        buffer.header.length = len(data)
        self._header("waveOutWrite", buffer)
        buffer.busy = True
        self.samples += len(data) // _SAMPLE_BYTES

    def start(self):
        self._call("waveOutRestart", self.handle)

    def position(self):
        cursor = _WaveTime(self._TIME_SAMPLES, 0, 0)
        self._call("waveOutGetPosition", self.handle, C.byref(cursor), C.sizeof(cursor))
        if cursor.unit == self._TIME_SAMPLES:
            return cursor.value / _RATE
        if cursor.unit == self._TIME_BYTES:
            return cursor.value / (_RATE * _SAMPLE_BYTES)
        if cursor.unit == self._TIME_MS:
            return cursor.value / 1000
        raise RuntimeError(f"Windows returned an unsupported audio clock format {cursor.unit}")

    def close(self):
        self.api.waveOutReset(self.handle)
        for buffer in self.buffers:
            self.api.waveOutUnprepareHeader(self.handle, C.byref(buffer.header), C.sizeof(buffer.header))
        self.api.waveOutClose(self.handle)


class LocalAudio:
    """Prepare audio off the UI thread and expose its media playback clock."""

    def __init__(self, path):
        self.ready = threading.Event()
        self.state = "preparing"
        self.error = ""
        self._start = threading.Event()
        self._stop = threading.Event()
        self._clock = None
        self._last_clock_read = 0.0
        self._thread = threading.Thread(target=self._run, args=(path,), name="local-audio", daemon=True)
        self._thread.start()

    def start(self):
        if self.state == "ready":
            self._start.set()

    def position(self):
        clock = self._clock
        if clock is None:
            return 0.0 if self._start.is_set() else None
        media, wall, limit = clock
        elapsed = max(0.0, time.perf_counter() - wall)
        candidate = media + elapsed if limit is None else min(media + elapsed, limit)
        self._last_clock_read = max(self._last_clock_read, candidate)
        return self._last_clock_read

    def close(self, *, wait=False):
        self._stop.set()
        if wait:
            self._thread.join(timeout=2)

    def _fill(self, decoder, device):
        for buffer in device.free_buffers():
            try:
                data = decoder.blocks.get_nowait()
            except Empty:
                break
            device.write(buffer, data)

    def _run(self, path):
        decoder = device = None
        try:
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if sys.platform != "win32" or not ffmpeg or not ffprobe:
                raise RuntimeError("Local audio requires Windows and FFmpeg with ffprobe in PATH")
            track = _media_audio(ffprobe, path)
            if self._stop.is_set():
                return
            if track is None:
                self.state = "silent"
                return
            decoder = _Decoder(ffmpeg, path, *track)
            device = _WaveOutput()
            deadline = time.perf_counter() + 5
            target = _CHUNK_BYTES * _DEVICE_BUFFERS // _SAMPLE_BYTES
            while not self._stop.is_set():
                self._fill(decoder, device)
                if decoder.error:
                    raise RuntimeError(decoder.error)
                if device.samples >= target or (decoder.done.is_set() and decoder.blocks.empty()):
                    break
                if time.perf_counter() >= deadline:
                    raise RuntimeError("Timed out while preparing the local soundtrack")
                self._stop.wait(_POLL_INTERVAL)
            if self._stop.is_set():
                return
            if not device.samples:
                self.state = "silent"
                return
            self.state = "ready"
            self.ready.set()
            while not self._stop.is_set() and not self._start.wait(0.01):
                pass
            if self._stop.is_set():
                return

            device.start()
            self.state = "playing"
            last_position = 0.0
            last_tick = time.perf_counter()
            while not self._stop.is_set():
                self._fill(decoder, device)
                if decoder.error:
                    raise RuntimeError(decoder.error)
                now = time.perf_counter()
                submitted = device.samples / _RATE
                position = min(device.position(), submitted)
                if position > last_position:
                    last_position, last_tick = position, now
                # Smooth the driver's short reporting intervals without
                # advancing indefinitely if the audio device runs out of data.
                limit = min(submitted, last_position + 0.024) if last_position else 0.0
                self._clock = last_position, last_tick, limit
                if (decoder.done.is_set() and decoder.blocks.empty()
                        and all(not b.busy for b in device.buffers)):
                    self._clock = submitted, now, None
                    self.state = "finished"
                    return
                if now - last_tick > (0.35 if last_position else 0.75):
                    raise RuntimeError("The Windows audio playback clock stopped")
                self._stop.wait(_POLL_INTERVAL)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if not self._stop.is_set():
                self.error = str(exc)
                position = self.position()
                if position is not None:
                    self._clock = position, time.perf_counter(), None
                self.state = "unavailable"
                if self._start.is_set():
                    print(
                        f"Local original audio stopped: {self.error}. "
                        "Check the Windows playback device and reopen the video.",
                        file=sys.stderr, flush=True,
                    )
        finally:
            self.ready.set()
            if device is not None:
                device.close()
            if decoder is not None:
                decoder.close()
            if self._stop.is_set() and self.state in {"preparing", "ready", "playing"}:
                self.state = "closed"
