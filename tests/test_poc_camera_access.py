"""POC cameras stay invisible to non-admin users, including wildcard users."""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import main  # noqa: E402
from config import (  # noqa: E402
    POC_FRAME_SKIP,
    frame_skip_for_camera,
    inference_device_for_camera,
)


def _user(role: str, allowed: list[str]) -> SimpleNamespace:
    return SimpleNamespace(role=role, allowed_cameras=allowed, feature_flags={})


def test_poc_cameras_are_denied_even_for_wildcard_user() -> None:
    user = _user("user", ["*"])
    for camera_id in main.config.POC_CAMERA_IDS:
        assert not main._camera_allowed(user, camera_id)


def test_admin_can_access_poc_cameras() -> None:
    admin = _user("admin", [])
    for camera_id in main.config.POC_CAMERA_IDS:
        assert main._camera_allowed(admin, camera_id)


def test_regular_camera_still_uses_user_allowlist() -> None:
    user = _user("user", ["prod-camera"])
    assert main._camera_allowed(user, "prod-camera")
    assert not main._camera_allowed(user, "other-camera")


def test_scope_excludes_poc_and_keeps_wildcard_regular_cameras() -> None:
    user = _user("user", ["*"])
    criteria = main._camera_scope_criteria(user, main.Detection.camera_id)
    assert len(criteria) == 1

    restricted = _user("user", ["prod-camera"])
    criteria = main._camera_scope_criteria(restricted, main.Detection.camera_id)
    assert len(criteria) == 2


def test_live_stream_denies_poc_cameras_to_non_admins() -> None:
    user = _user("user", ["*"])
    for camera_id in main.config.POC_CAMERA_IDS:
        with pytest.raises(main.HTTPException) as error:
            asyncio.run(main.stream_mjpeg(camera_id, user))
        assert error.value.status_code == 403


def test_poc_sampling_budget_is_separate_from_existing_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert frame_skip_for_camera("regular-camera") == main.config.FRAME_SKIP
    monkeypatch.setattr(main.config, "POC_EDGE_ENABLED", True)
    for camera_id in main.config.POC_CAMERA_IDS:
        assert frame_skip_for_camera(camera_id) == POC_FRAME_SKIP


def test_poc_device_override_does_not_move_existing_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.config, "POC_EDGE_ENABLED", True)
    monkeypatch.setattr(main.config, "POC_INFERENCE_DEVICE", "cpu")
    assert inference_device_for_camera("poc-camera-2") == "cpu"
    assert inference_device_for_camera("poc-camera-1") == "cpu"
    monkeypatch.setattr(main.config, "INFERENCE_DEVICE", "mps")
    assert inference_device_for_camera("場域丙") == "mps"


def test_poc_burst_queue_is_larger_but_existing_queue_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import services.stream as stream_module  # noqa: E402

    monkeypatch.setattr(main.config, "POC_EDGE_ENABLED", True)
    monkeypatch.setattr(stream_module, "POC_EDGE_ENABLED", True)
    poc = stream_module.CameraStream("poc-camera-2", "https://example.test/live")
    regular = stream_module.CameraStream("場域丙", "rtsp://example.test/live")
    assert poc._pending.maxlen == main.config.POC_FRAME_QUEUE_MAX
    assert regular._pending.maxlen == 16
