"""RTSP stream capture and MJPEG serving.

Each camera runs in its own background thread. A shared `latest_frame` dict
(keyed by camera_id) is used by both the MJPEG endpoint and the inference
worker so the RTSP connection is opened only once.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional
from urllib.parse import urlparse

import os

# RTSP over TCP by default: the ffmpeg default (UDP) loses packets under load,
# which corrupts the h264 stream and shows up as decode warnings and
# lost-frame reconnects. cht.bovia accepts TCP (verified 4/4 cameras,
# delivery unchanged at 25-30 fps). Overridable via the environment.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

# Ceiling for the reconnect backoff. Five minutes keeps a permanently dead
# source at ~12 open attempts an hour instead of 360, while still recovering a
# camera that comes back within the operator's attention span.
_OPEN_BACKOFF_MAX_SECONDS = 300.0

from config import (
    CAPTURE_FPS_WINDOW_SECONDS,
    IDLE_GATE_STDDEV,
    POC_CAMERA_IDS,
    POC_EDGE_ENABLED,
    POC_FRAME_QUEUE_MAX,
    STREAM_FPS,
)

logger = logging.getLogger(__name__)


def _is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().split(":", 1)[0]
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _resolve_capture_url(url: str) -> Optional[str]:
    """Resolve a YouTube watch page to one bounded live HLS rendition.

    OpenCV cannot open a watch page itself and YouTube's signed HLS URLs expire,
    so resolution happens only on the source's capture/reconnect thread. A
    normal RTSP or HTTP camera returns unchanged and never imports yt-dlp.
    """
    if not _is_youtube_url(url):
        return url
    try:
        from yt_dlp import YoutubeDL

        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        formats = [
            item for item in (info.get("formats") or [])
            if item.get("url") and item.get("protocol", "").startswith("m3u8")
        ]
        if not formats:
            logger.warning("YouTube source resolved without an HLS format")
            return None
        bounded = [item for item in formats if item.get("height") and item["height"] <= 720]
        chosen = max(
            bounded or formats,
            key=lambda item: (item.get("height") or 0, item.get("width") or 0),
        )
        logger.info(
            "YouTube source resolved to HLS (%sx%s)",
            chosen.get("width") or "?", chosen.get("height") or "?",
        )
        return chosen["url"]
    except Exception as exc:
        logger.warning("YouTube source resolution failed: %s", exc)
        return None


class CameraStream:
    """Manages a single RTSP camera connection in a background thread."""

    # Vitality: a working camera shows pixel variance. Pure black / signal-
    # lost feeds sit at std ~0. A real low-light scene still has grain (>5).
    _VITALITY_STDDEV_THRESHOLD = 5.0
    _VITALITY_CHECK_INTERVAL = 1.0  # seconds between vitality fingerprint checks
    _VITALITY_STALE_AFTER = 15.0  # consider dead if no meaningful frame within N seconds

    def __init__(self, camera_id: str, rtsp_url: str) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url

        self._frame: Optional[np.ndarray] = None
        self._seq = 0  # increments per captured frame, so consumers can tell
                       # a fresh frame from the one they already processed
        # Capture-side observability: production workers found no new frame on
        # 93% of GPU turns while fresh probe sessions delivered 30 fps, so the
        # capture rate has to be visible per camera to tell a starved capture
        # thread from a slow consumer.
        # Keep enough samples for the capture-rate window. A 90-frame window
        # was correct for steady RTSP but turned an HLS burst into nonsense:
        # YouTube could deliver those 90 frames in ~35 ms and the dashboard
        # reported 2,951 FPS. The gap diagnostic still uses the most recent
        # 90 intervals; the rate itself uses a wall-clock window below.
        self._cap_times: deque[float] = deque(maxlen=1200)
        self.reconnects = 0
        # Consecutive failures to obtain a usable capture. Drives the backoff
        # in _capture_loop; reset the moment a stream opens.
        self._open_failures = 0
        # Interrupts that backoff. stop() only joins for 5 s, and the backoff
        # sleeps for up to 300 — without this, releasing a camera that is
        # mid-retry leaves the thread alive until its sleep runs out.
        self._wake = threading.Event()
        self._lock = threading.Lock()
        # Wakes consumers exactly once per captured frame. Polling with short
        # sleeps was untrustworthy under GIL contention: a nominal 2 ms sleep
        # can take 10-15 ms to actually resume, and the accounting recorded
        # the nominal value — 200 ms/frame of real wall time went unbooked.
        self._new_frame = threading.Condition(self._lock)
        # RTSP delivery is BURSTY, not steady: measured inter-frame gaps of
        # p50 ~1 ms with 200-3000 ms silences between bursts. A latest-frame
        # consumer can only act once per burst (3-4 fps against a 33 fps
        # average), so every frame is queued and the detector drains the burst
        # during the silence. Bounded: overflow drops the OLDEST entry, so a
        # lagging consumer loses history, never freshness.
        queue_max = (
            POC_FRAME_QUEUE_MAX
            if POC_EDGE_ENABLED and camera_id in POC_CAMERA_IDS
            else 16
        )
        self._pending: deque[tuple[int, np.ndarray, float]] = deque(maxlen=queue_max)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        # Frame vitality state — distinguishes "RTSP connected" from
        # "RTSP delivers a meaningful, non-black image". Updated by the
        # capture loop at ~1 Hz.
        self._last_vitality_check: float = 0.0
        self._last_meaningful_frame_ts: float = 0.0
        # Separate, stricter timestamp for the inference gate: the last frame
        # that carried ANY image at all (std above IDLE_GATE_STDDEV, which sits
        # below the vitality threshold). Kept apart from the vitality stamp so
        # the UI's "dark" verdict and the decision to stop inferring can never
        # be tightened by accident — see `is_blank`.
        self._last_lit_frame_ts: float = 0.0
        self._frame_stddev: float = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera %s: capture thread started (%s)", self.camera_id, self.rtsp_url)

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Camera %s: capture thread stopped", self.camera_id)

    # ── frame access (thread-safe) ───────────────────────────────────────

    @property
    def frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def frame_after(self, seq: int) -> tuple[Optional[np.ndarray], int]:
        """Return (frame, seq) only if a frame newer than `seq` has arrived.

        Lets the detector process every captured frame exactly once instead of
        polling on a fixed sleep: no duplicate inference when it runs ahead of
        the camera, and no idle GPU while it waits out a timer when it doesn't.
        """
        with self._lock:
            if self._frame is None or self._seq <= seq:
                return None, self._seq
            return self._frame.copy(), self._seq

    @property
    def connected(self) -> bool:
        return self._connected

    def next_frame(self, after_seq: int, timeout: float) -> tuple[Optional[np.ndarray], int, float]:
        """Pop the oldest un-consumed frame newer than `after_seq`.

        Returns (frame, seq, capture_ts), or (None, after_seq, 0.0) on
        timeout. FIFO through the burst queue: the consumer processes every
        frame a burst delivered, in order, instead of sampling one per burst.
        Single consumer per camera (its DetectorWorker).
        """
        deadline = time.monotonic() + timeout
        with self._new_frame:
            while True:
                while self._pending and self._pending[0][0] <= after_seq:
                    self._pending.popleft()
                if self._pending:
                    seq, frame, ts = self._pending.popleft()
                    return frame, seq, ts
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._new_frame.wait(remaining):
                    return None, after_seq, 0.0

    @property
    def capture_gaps_ms(self) -> dict:
        """Percentiles of inter-frame arrival gaps over the recent window.

        Distinguishes steady 30 ms delivery from bursty delivery (packets of
        6-7 frames every ~200 ms) — a latest-frame-only consumer sees the
        same average rate either way but can only act once per burst.
        """
        times = list(self._cap_times)
        if len(times) < 10:
            return {}
        gaps = sorted((b - a) * 1000 for a, b in zip(times, times[1:]))
        n = len(gaps)
        return {
            "p50": round(gaps[n // 2], 1),
            "p90": round(gaps[int(n * 0.9)], 1),
            "max": round(gaps[-1], 1),
        }

    @property
    def capture_fps(self) -> float:
        """Delivered frames/sec over a wall-clock window (0 when stale).

        HLS and some RTSP decoders release frames in bursts. Dividing the last
        few frames by their inter-arrival span measures decoder burst speed,
        not source delivery speed. The denominator is deliberately the fixed
        wall-clock window, so a burst cannot produce a thousands-of-FPS value.
        """
        times = list(self._cap_times)
        if len(times) < 2:
            return 0.0
        now = time.time()
        window = CAPTURE_FPS_WINDOW_SECONDS
        if (now - times[-1]) > window:
            return 0.0
        cutoff = now - window
        recent = [stamp for stamp in times if stamp >= cutoff]
        if len(recent) < 2:
            return 0.0
        return round((len(recent) - 1) / window, 2)

    @property
    def is_alive(self) -> bool:
        """Stream is actually delivering meaningful (non-black) frames.

        True only if RTSP is connected AND we have seen a recent frame whose
        pixel variance exceeded the vitality threshold. A camera that shows
        a frozen black frame (lens covered, lost signal, etc.) returns False
        even though `connected` is True.

        File sources are always considered alive once they connect — the
        loop is deterministic, no need for vitality checks.
        """
        if not self._connected:
            return False
        if self._is_file_source:
            return True
        if self._last_meaningful_frame_ts == 0.0:
            return False  # never seen a meaningful frame yet
        return time.time() - self._last_meaningful_frame_ts < self._VITALITY_STALE_AFTER

    @property
    def is_blank(self) -> bool:
        """Frames carry no image at all — safe to sample at a heartbeat rate.

        Strictly stronger than `not is_alive`: the gate threshold sits BELOW
        the vitality threshold, so a dim-but-real scene the dashboard already
        shows as dark keeps full-rate inference. Blackness must also PERSIST
        for `_VITALITY_STALE_AFTER` before the gate closes, while a single lit
        frame re-opens it instantly — hysteresis in the direction that costs
        GPU rather than the direction that costs detections.

        False for file sources (deterministic test loops) and whenever the
        stream is down, so a disconnected camera keeps its existing path.
        """
        if IDLE_GATE_STDDEV <= 0 or not self._connected or self._is_file_source:
            return False
        # The stamp is primed at connect time, so a stream that has never shown
        # anything still gets a full stale-window of full-rate inference before
        # the gate closes — "no evidence yet" must never read as "blank".
        return time.time() - self._last_lit_frame_ts >= self._VITALITY_STALE_AFTER

    @property
    def frame_stddev(self) -> float:
        """Most recent frame pixel standard deviation (vitality fingerprint)."""
        return self._frame_stddev

    @property
    def _is_file_source(self) -> bool:
        """Detect a local video file, not an RTSP/HTTP live source."""
        return urlparse(self.rtsp_url).scheme not in {"rtsp", "http", "https"}

    # ── internal capture loop ────────────────────────────────────────────

    def _backoff_delay(self, base: float) -> float:
        """Seconds to wait before the next open attempt.

        A source that is down for a minute is worth retrying often; one that is
        down permanently is not. Two YouTube cameras added through the admin UI
        started failing YouTube's bot check, and the old fixed 10 s retry spawned
        a yt-dlp subprocess every 10 seconds each, forever — together with the
        always-on RTSP decoding that pinned the 0.5-vCPU api container at 100%
        CPU for days and made the dashboard return "Failed to fetch" at random
        (2026-09-20). Doubling up to a five-minute ceiling keeps a blip cheap to
        recover from and makes a dead source cost almost nothing.
        """
        return min(base * (2 ** min(self._open_failures, 6)), _OPEN_BACKOFF_MAX_SECONDS)

    def _capture_loop(self) -> None:
        while self._running:
            capture_url = _resolve_capture_url(self.rtsp_url)
            if not capture_url:
                self._connected = False
                self._open_failures += 1
                delay = self._backoff_delay(10)
                logger.warning(
                    "Camera %s: source did not resolve (attempt %d), retrying in %.0fs …",
                    self.camera_id, self._open_failures, delay,
                )
                self._wake.wait(delay)
                continue
            cap = cv2.VideoCapture(capture_url)
            if not cap.isOpened():
                self._connected = False
                self._open_failures += 1
                delay = self._backoff_delay(5)
                logger.warning(
                    "Camera %s: failed to open stream (attempt %d), retrying in %.0fs …",
                    self.camera_id, self._open_failures, delay,
                )
                self._wake.wait(delay)
                continue

            self._open_failures = 0
            self._connected = True
            # Prime the gate stamp on every (re)connect: the idle gate may only
            # close after a full stale-window of measured blackness, never
            # because the stream is young and has not been fingerprinted yet.
            self._last_lit_frame_ts = time.time()
            is_file = self._is_file_source
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_interval = 1.0 / fps if is_file else 0

            logger.info(
                "Camera %s: connected (%s, %.1f fps)",
                self.camera_id,
                "file" if is_file else "rtsp",
                fps,
            )

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    if is_file:
                        # Loop video files back to the beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.debug("Camera %s: looping video file", self.camera_id)
                        continue
                    else:
                        logger.warning("Camera %s: lost frame, reconnecting …", self.camera_id)
                        self.reconnects += 1
                        self._connected = False
                        break
                now = time.time()
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._pending.append((self._seq, frame, now))
                    self._new_frame.notify_all()
                self._cap_times.append(now)

                # Vitality fingerprint (throttled to ~1 Hz). A working
                # camera shows non-zero pixel variance even at night; a
                # frozen black frame sits at std ~0.
                self._check_vitality(frame)

                # Throttle file playback to natural FPS
                if is_file and frame_interval > 0:
                    time.sleep(frame_interval)

            cap.release()

        self._connected = False

    def _check_vitality(self, frame: np.ndarray) -> None:
        """Compute frame pixel std on a downsampled copy (cheap, ~1ms).

        Updates `_last_meaningful_frame_ts` whenever variance crosses the
        threshold, so `is_alive` reports `False` once frames go flat-black,
        and `_last_lit_frame_ts` on the lower gate threshold, which is what
        `is_blank` (and therefore the detector's idle gate) reads.
        """
        now = time.time()
        if now - self._last_vitality_check < self._VITALITY_CHECK_INTERVAL:
            return
        self._last_vitality_check = now
        try:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            self._frame_stddev = float(np.std(small))
            if self._frame_stddev > self._VITALITY_STDDEV_THRESHOLD:
                self._last_meaningful_frame_ts = now
            if self._frame_stddev > IDLE_GATE_STDDEV:
                self._last_lit_frame_ts = now
        except Exception:
            # Resize / std failure should never break the capture loop.
            pass


class StreamManager:
    """Registry of all active camera streams."""

    def __init__(self) -> None:
        self._streams: dict[str, CameraStream] = {}

    def add(self, camera_id: str, rtsp_url: str) -> CameraStream:
        if camera_id in self._streams:
            return self._streams[camera_id]
        stream = CameraStream(camera_id, rtsp_url)
        stream.start()
        self._streams[camera_id] = stream
        return stream

    def get(self, camera_id: str) -> Optional[CameraStream]:
        return self._streams.get(camera_id)

    def remove(self, camera_id: str) -> None:
        stream = self._streams.pop(camera_id, None)
        if stream:
            stream.stop()

    def stop_all(self) -> None:
        for stream in self._streams.values():
            stream.stop()
        self._streams.clear()

    def list_ids(self) -> list[str]:
        return list(self._streams.keys())

    # ── MJPEG generator ──────────────────────────────────────────────────

    def mjpeg_generator(self, camera_id: str):
        """Yields MJPEG multipart frames for an HTTP streaming response."""
        stream = self.get(camera_id)
        if stream is None:
            return

        interval = 1.0 / STREAM_FPS
        while True:
            frame = stream.frame
            if frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
            time.sleep(interval)


# Module-level singleton
stream_manager = StreamManager()
