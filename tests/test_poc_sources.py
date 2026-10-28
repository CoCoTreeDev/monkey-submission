"""POC capture source resolution is isolated from regular camera URLs."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from services.stream import (  # noqa: E402
    CameraStream,
    _is_youtube_url,
    _resolve_capture_url,
)


def test_regular_camera_urls_are_not_resolved() -> None:
    url = "rtsp://camera.example/live"
    assert not _is_youtube_url(url)
    assert _resolve_capture_url(url) == url


def test_youtube_resolution_picks_highest_hls_under_720p(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download=False):
            assert download is False
            return {
                "formats": [
                    {"url": "audio", "protocol": "http", "height": 720},
                    {"url": "hls-1080", "protocol": "m3u8_native", "width": 1920, "height": 1080},
                    {"url": "hls-480", "protocol": "m3u8_native", "width": 854, "height": 480},
                    {"url": "hls-720", "protocol": "m3u8_native", "width": 1280, "height": 720},
                ]
            }

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    assert _is_youtube_url("https://www.youtube.com/watch?v=fixture-video")
    assert _resolve_capture_url("https://www.youtube.com/watch?v=fixture-video") == "hls-720"


def test_http_poc_source_is_treated_as_live_not_a_looping_file() -> None:
    stream = CameraStream("poc-camera-1", "https://cctv.example/live")
    assert stream._is_file_source is False
