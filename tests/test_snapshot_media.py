"""Exercise browser media requests through the authenticated snapshot route."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import main  # noqa: E402
from services.storage import LocalStorage  # noqa: E402

PAYLOAD = bytes(range(256)) * 1024


@pytest.fixture
def client(tmp_path, monkeypatch):
    directory = tmp_path / "snapshots"
    store = LocalStorage(directory)
    store.upload("clip.mp4", PAYLOAD)
    store.upload("still.jpg", b"jpeg")
    (directory / "rebuild-review").mkdir()
    store.upload("rebuild-review/場域甲.mp4", PAYLOAD)
    monkeypatch.setattr(main, "storage", store)
    # No lifespan: media requests need neither cameras nor inference workers.
    browser = TestClient(main.app)
    try:
        browser.headers["Authorization"] = "Bearer " + main._sign_token(1, "user")
        yield browser
    finally:
        browser.close()


@pytest.mark.parametrize(
    ("requested", "start", "end"),
    [("bytes=0-1", 0, 2), ("bytes=150000-215535", 150000, 215536),
     ("bytes=250000-", 250000, len(PAYLOAD)),
     ("bytes=-32", len(PAYLOAD) - 32, len(PAYLOAD))],
)
def test_video_range_sends_only_requested_bytes(client, requested, start, end):
    response = client.get("/api/snapshots/clip.mp4", headers={"Range": requested})
    assert response.status_code == 206
    assert response.content == PAYLOAD[start:end]
    assert response.headers["content-range"] == f"bytes {start}-{end - 1}/{len(PAYLOAD)}"
    assert response.headers["content-length"] == str(end - start)
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-type"] == "video/mp4"


def test_full_video_and_head_describe_the_same_file(client):
    response = client.get("/api/snapshots/clip.mp4")
    head = client.head("/api/snapshots/clip.mp4")
    assert response.status_code == head.status_code == 200
    assert response.content == PAYLOAD
    assert head.content == b""
    for name in ("content-length", "accept-ranges", "etag", "last-modified"):
        assert head.headers[name] == response.headers[name]
    assert head.headers["content-length"] == str(len(PAYLOAD))


def test_range_head_has_no_body(client):
    response = client.head("/api/snapshots/clip.mp4", headers={"Range": "bytes=0-1"})
    assert response.status_code == 206
    assert response.content == b""
    assert response.headers["content-length"] == "2"


def test_head_does_not_open_or_stream_the_file(client, monkeypatch):
    import anyio

    def unexpected_read(*args, **kwargs):
        pytest.fail("HEAD must only stat the file")

    monkeypatch.setattr(anyio, "open_file", unexpected_read)
    monkeypatch.setattr(main.storage, "download_stream", unexpected_read)
    response = client.head("/api/snapshots/clip.mp4")
    assert response.status_code == 200
    assert response.headers["content-length"] == str(len(PAYLOAD))


def test_unsatisfiable_range_is_not_a_full_download(client):
    response = client.get("/api/snapshots/clip.mp4", headers={"Range": "bytes=999999-"})
    assert response.status_code == 416
    assert response.headers["content-range"].endswith(f"/{len(PAYLOAD)}")
    assert response.content != PAYLOAD


def test_if_range_uses_current_file_validator(client):
    etag = client.head("/api/snapshots/clip.mp4").headers["etag"]
    response = client.get("/api/snapshots/clip.mp4", headers={"Range": "bytes=0-1", "If-Range": etag})
    assert response.status_code == 206
    assert response.content == PAYLOAD[:2]
    changed = client.get("/api/snapshots/clip.mp4", headers={"Range": "bytes=0-1", "If-Range": '"stale"'})
    assert changed.status_code == 200
    assert changed.content == PAYLOAD


def test_nested_review_clip_and_still_remain_available(client):
    clip = client.get("/api/snapshots/rebuild-review/場域甲.mp4", headers={"Range": "bytes=0-1"})
    assert clip.status_code == 206
    assert clip.content == PAYLOAD[:2]
    still = client.get("/api/snapshots/still.jpg")
    assert still.status_code == 200
    assert still.content == b"jpeg"
    assert still.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("path", ["missing.mp4", "rebuild-review"])
def test_missing_file_or_directory_is_404(client, path):
    assert client.get(f"/api/snapshots/{path}").status_code == 404


@pytest.mark.parametrize("path", ["other/clip.mp4", "rebuild-review/%2E%2E/clip.mp4", "bad%5Cclip.mp4"])
def test_invalid_path_is_rejected(client, path):
    assert client.get(f"/api/snapshots/{path}").status_code == 400


def test_media_still_requires_authentication(client):
    client.headers.pop("Authorization")
    for method in (client.get, client.head):
        assert method("/api/snapshots/clip.mp4", headers={"Range": "bytes=0-1"}).status_code == 401
    token = main._sign_token(1, "user")
    assert client.get("/api/snapshots/clip.mp4", params={"token": token}, headers={"Range": "bytes=0-1"}).status_code == 206


def test_blob_redirect_is_preserved(client, monkeypatch):
    monkeypatch.setattr(main, "storage", SimpleNamespace(
        exists=lambda name: True,
        signed_url=lambda name: "https://media.example/clip.mp4?signature=test",
    ))
    response = client.get("/api/snapshots/clip.mp4", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "https://media.example/clip.mp4?signature=test"
    assert response.headers["cache-control"] == "private, max-age=2700"


def test_storage_without_a_signed_url_keeps_streaming_fallback(client, monkeypatch):
    monkeypatch.setattr(main, "storage", SimpleNamespace(
        exists=lambda name: True,
        signed_url=lambda name: None,
        download_stream=lambda name: iter([b"first", b"second"]),
    ))
    response = client.get("/api/snapshots/clip.mp4")
    assert response.status_code == 200
    assert response.content == b"firstsecond"
