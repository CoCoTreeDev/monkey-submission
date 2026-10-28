"""A single-node deployment must still alert when a camera goes blind.

The node that runs the detectors is also the node that runs the watchdog, so
its own heartbeat has to carry the per-camera stats and the watchdog has to
check it — the stall alert exists because 場域甲 sat blind for 62 hours.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import main  # noqa: E402


def _beat(cams: list[dict]) -> dict:
    return {"last_seen_ts": 0.0, "alerted": False, "payload": {"cameras": cams}}


def _run(coro):
    return asyncio.run(coro)


def test_local_node_cameras_are_checked_for_stalls(monkeypatch) -> None:
    sent: list[tuple[str, str]] = []

    async def fake_alert(title, body, **kw):
        sent.append((title, body))

    monkeypatch.setattr(main, "send_system_alert", fake_alert)
    monkeypatch.setattr(main.config, "CAMERA_STALL_ALERT_SECONDS", 60)

    st = _beat([
        {"camera_id": "場域甲", "stream_connected": True,
         "seconds_since_last": 9999, "worker_alive": True},
    ])
    _run(main._check_stalled(main._LOCAL_NODE, st))

    assert len(sent) == 1, "a blind camera on the local node must still alert"
    assert "場域甲" in sent[0][1]
    assert "停止偵測" in sent[0][0]


def test_a_stall_alerts_once_then_reports_recovery(monkeypatch) -> None:
    sent: list[str] = []

    async def fake_alert(title, body, **kw):
        sent.append(title)

    monkeypatch.setattr(main, "send_system_alert", fake_alert)
    monkeypatch.setattr(main.config, "CAMERA_STALL_ALERT_SECONDS", 60)

    stalled = _beat([{"camera_id": "場域丙", "stream_connected": True,
                      "seconds_since_last": 9999, "worker_alive": True}])
    _run(main._check_stalled("node", stalled))
    _run(main._check_stalled("node", stalled))
    assert sent == ["🔴 攝影機停止偵測"], "must not re-alert every tick"

    stalled["payload"]["cameras"][0]["seconds_since_last"] = 1
    _run(main._check_stalled("node", stalled))
    assert sent[-1] == "🟢 攝影機恢復偵測"


def test_healthy_cameras_are_silent(monkeypatch) -> None:
    sent: list[str] = []

    async def fake_alert(title, body, **kw):
        sent.append(title)

    monkeypatch.setattr(main, "send_system_alert", fake_alert)
    monkeypatch.setattr(main.config, "CAMERA_STALL_ALERT_SECONDS", 60)

    st = _beat([{"camera_id": "場域丁", "stream_connected": True,
                 "seconds_since_last": 0.4, "worker_alive": True}])
    _run(main._check_stalled("node", st))
    assert sent == []


def test_the_node_heartbeat_reports_cameras() -> None:
    import inspect

    src = inspect.getsource(main._node_heartbeat_loop)
    assert "detector_manager.inference_stats()" in src
