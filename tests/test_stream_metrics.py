"""Capture-rate metrics must not mistake decoder bursts for source FPS."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import services.stream as stream_module  # noqa: E402


def test_capture_fps_uses_wall_clock_window_for_hls_burst(
    monkeypatch,
) -> None:
    now = 100.0
    stream = stream_module.CameraStream("poc-camera-2", "https://example.test/live")
    # One hundred frames released in 40 ms is a decoder burst, not 2,500 FPS.
    stream._cap_times.extend(99.96 + i * 0.0004 for i in range(100))
    monkeypatch.setattr(stream_module.time, "time", lambda: now)

    assert stream.capture_fps == 9.9


def test_capture_fps_counts_a_sustained_source_rate(
    monkeypatch,
) -> None:
    now = 100.0
    stream = stream_module.CameraStream("regular", "rtsp://example.test/live")
    stream._cap_times.extend(90.0 + i / 30.0 for i in range(301))
    monkeypatch.setattr(stream_module.time, "time", lambda: now)

    assert stream.capture_fps == 30.0


def test_open_backoff_grows_and_caps() -> None:
    """A permanently dead source must stop costing a subprocess every 10 s.

    Two YouTube cameras started failing YouTube's bot check and the old fixed
    retry spawned yt-dlp forever, pinning the api container's CPU (2026-09-20).
    """
    stream = stream_module.CameraStream("dead", "https://www.youtube.com/watch?v=x")

    # First failure retries promptly; the delay then doubles.
    delays = []
    for _ in range(9):
        stream._open_failures += 1
        delays.append(stream._backoff_delay(10))

    assert delays[0] == 20      # 10 * 2**1
    assert delays[1] == 40
    assert delays[2] == 80
    assert delays == sorted(delays), "backoff must never shrink while failing"
    assert max(delays) == stream_module._OPEN_BACKOFF_MAX_SECONDS
    # Capped, not unbounded: an hour of downtime is ~12 attempts, not 360.
    assert delays[-1] == 300


def test_open_backoff_resets_after_a_success() -> None:
    stream = stream_module.CameraStream("flappy", "rtsp://example.test/live")
    stream._open_failures = 5
    assert stream._backoff_delay(5) > 5

    # _capture_loop clears the counter the moment a capture opens.
    stream._open_failures = 0
    assert stream._backoff_delay(5) == 5
