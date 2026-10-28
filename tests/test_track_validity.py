"""Regression tests for rejecting one-frame-like monkey tracks."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from track_policy import is_substantial_track  # noqa: E402
from services.detector import (  # noqa: E402
    DetectorWorker,
    _FrameBuffer,
    _PendingVisit,
    _TrackState,
)
import services.detector as detector_module  # noqa: E402


def test_track_needs_both_hits_and_observed_time() -> None:
    assert not is_substantial_track(1, 2.0, min_hits=3, min_seconds=0.5)
    assert not is_substantial_track(3, 0.2, min_hits=3, min_seconds=0.5)
    assert is_substantial_track(3, 0.5, min_hits=3, min_seconds=0.5)


def test_rebuild_does_not_keep_two_adjacent_boxes() -> None:
    # Two adjacent frames are exactly the false-positive shape seen in the
    # production reference run: technically two boxes, visually one instant.
    assert not is_substantial_track(2, 0.1, min_hits=3, min_seconds=0.5)


def _track(times: list[float]) -> _TrackState:
    track = _TrackState(track_id=1, camera_id="cam", frame_idx=0)
    for frame_idx, captured_ts in enumerate(times):
        track.update(
            (20, 20), 0.9, (10.0, 10.0, 20.0, 20.0), frame_idx,
            captured_ts=captured_ts,
        )
    return track


def test_live_finalise_rejects_short_track(monkeypatch) -> None:
    worker = DetectorWorker("cam")
    finalised: list[object] = []
    monkeypatch.setattr(
        worker,
        "_finalise_group",
        lambda group, still_active, frame_buf: finalised.append(group),
    )

    worker._finalise_batch([_track([10.0, 10.1])], {}, _FrameBuffer())
    assert finalised == []

    worker._finalise_batch([_track([10.0, 10.25, 10.5])], {}, _FrameBuffer())
    assert len(finalised) == 1


def test_live_render_does_not_emit_without_trail(monkeypatch) -> None:
    worker = DetectorWorker("cam")
    emitted: list[object] = []
    worker._emit_detection = lambda *args, **kwargs: emitted.append(args)
    monkeypatch.setattr(
        detector_module,
        "_generate_combined_trail_video",
        lambda *args: ("", [], 0.0),
    )

    visit = _PendingVisit(
        "cam", [(_track([10.0, 10.25, 10.5]), None)], [],
        _FrameBuffer(), time.monotonic(),
    )
    DetectorWorker._render_visit(worker, visit)

    assert emitted == []
