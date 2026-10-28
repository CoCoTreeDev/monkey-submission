"""Regression tests for the trail video's fixed-rate timeline."""

from __future__ import annotations

import sys

sys.path.insert(0, "backend")

import pytest

from services.detector import _resample_trail_frames  # noqa: E402
from services.detector import _clip_offsets  # noqa: E402


@pytest.mark.parametrize("encoder", ["live", "rebuild"])
def test_mp4_seek_never_needs_more_than_two_seconds_of_frames(tmp_path, encoder) -> None:
    import json
    import shutil
    import subprocess

    import numpy as np
    from PIL import Image

    from services.detector import _encode_mp4
    from rebuild_clips import encode

    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe unavailable")
    # A static scene prevents incidental scene-cut keyframes from hiding the
    # long-GOP regression. Include the final span after the last keyframe.
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    path = tmp_path / "seek.mp4"
    fps = 24 if encoder == "live" else 10
    if encoder == "live":
        path.write_bytes(_encode_mp4([Image.fromarray(frame)] * (fps * 5),
                                     [i / fps for i in range(fps * 5)]))
    else:
        encode([frame] * (fps * 5), fps, path)
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-skip_frame", "nokey", "-select_streams", "v:0",
        "-show_entries", "frame=pts_time:format=duration", "-of", "json", str(path),
    ]))
    times = [float(frame["pts_time"]) for frame in result["frames"]]
    times.append(float(result["format"]["duration"]))
    assert times[0] == 0
    assert times[-1] == pytest.approx(5)
    assert max(b - a for a, b in zip(times, times[1:])) <= 2 + 1 / fps


class _Track:
    first_seen_ts = 10.1
    last_seen_ts = 10.1
    frame_indices: list[int] = []


def test_resampling_keeps_every_source_frame_and_fills_to_24_fps() -> None:
    frames = [object(), object(), object()]
    output, timestamps, fps = _resample_trail_frames(
        frames, [100.0, 100.2, 100.4], target_fps=24.0,
    )

    assert fps == 24.0
    assert len(output) == 11
    assert len(timestamps) == len(output)
    assert timestamps == [100.0] * 5 + [100.2] * 5 + [100.4]
    assert all(frame in output for frame in frames)
    assert output[0] is frames[0]
    assert output[5] is frames[1]
    assert output[10] is frames[2]


def test_resampling_preserves_dense_frames_in_order() -> None:
    frames = [object() for _ in range(5)]
    output, _, fps = _resample_trail_frames(
        frames, [10.0, 10.001, 10.002, 10.003, 10.004], target_fps=24.0,
    )

    assert fps == 24.0
    positions = [output.index(frame) for frame in frames]
    assert positions == sorted(positions)
    assert all(frame in output for frame in frames)


def test_offsets_follow_source_slots_after_dense_frame_collision() -> None:
    """A dense burst must not make the last selected frame seek early."""
    frames = [object() for _ in range(5)]
    output, timestamps, fps = _resample_trail_frames(
        frames, [10.0, 10.001, 10.002, 10.003, 10.1], target_fps=24.0,
    )

    # The fifth source frame is forced to output slot 4 to preserve ordering:
    # four earlier frames already occupied slots 0..3. A wall-clock bisect on
    # a regular playback axis incorrectly maps 10.1 to slot 3.
    assert output[4] is frames[4]
    offsets = _clip_offsets(_Track(), timestamps, fps)
    assert offsets["clip_offset_seconds"] == pytest.approx(4 / 24, abs=0.001)


def test_long_wall_clock_gap_stays_bounded_without_dropping_selected_frames() -> None:
    frames = [object() for _ in range(150)]
    timestamps = [100.0 + i * 1.0 for i in range(len(frames))]

    output, _, fps = _resample_trail_frames(frames, timestamps, target_fps=24.0)

    assert fps == 24.0
    assert len(output) == 600
    assert all(frame in output for frame in frames)
