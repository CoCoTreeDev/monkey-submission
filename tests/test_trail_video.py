"""Trail MP4 contract — no model, camera, storage, or network required.

The production requirement is deliberately stronger than "ffmpeg returned a
file": the clip must be H.264 MP4, fast-start, and retain every selected frame.
Run with ``uv run --project backend python tests/test_trail_video.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "backend")

from services import detector as det_mod  # noqa: E402


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {label}")
        return
    print(f"  FAIL  {label}: {detail}")
    raise SystemExit(1)


ffmpeg = shutil.which(det_mod.FFMPEG_BINARY)
ffprobe = shutil.which("ffprobe")
if not ffmpeg or not ffprobe:
    # Local test environments without ffmpeg still exercise the critical
    # fail-open behaviour below; production images install ffmpeg explicitly.
    print("SKIP real MP4 probe: ffmpeg/ffprobe unavailable")
else:
    print("Trail MP4 encoding")
    frames = []
    for i in range(12):
        frame = np.zeros((72, 128, 3), dtype=np.uint8)
        frame[:, : 12 + i * 8] = (i * 19, 180, 255 - i * 13)
        frames.append(Image.fromarray(frame, "RGB"))
    timestamps = [1000.0 + i * 0.04 for i in range(len(frames))]

    mp4 = det_mod._encode_mp4(frames, timestamps)
    check("encoder returns an MP4", mp4 is not None and len(mp4) > 0,
          f"bytes={len(mp4 or b'')}")
    assert mp4 is not None
    check("faststart places moov before media data",
          0 <= mp4.find(b"moov") < mp4.find(b"mdat"),
          f"moov={mp4.find(b'moov')}, mdat={mp4.find(b'mdat')}")

    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,nb_read_frames,avg_frame_rate",
            "-of", "default=noprint_wrappers=1", "pipe:0",
        ],
        input=mp4,
        capture_output=True,
        check=True,
    ).stdout.decode()
    values = dict(line.split("=", 1) for line in probe.splitlines() if "=" in line)
    check("uses H.264", values.get("codec_name") == "h264", probe)
    check("uses the configured 24 FPS timeline", values.get("avg_frame_rate") == "24/1", probe)
    check("keeps every selected frame", values.get("nb_read_frames") == str(len(frames)), probe)


print("Trail MP4 fail-open")
saved_binary = det_mod.FFMPEG_BINARY
try:
    det_mod.FFMPEG_BINARY = "ffmpeg-not-installed-for-test"
    missing = det_mod._encode_mp4(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))], [0.0, 0.04]
    )
    check("a missing encoder loses only the optional clip", missing is None, repr(missing))
finally:
    det_mod.FFMPEG_BINARY = saved_binary


# The encoder must be FOUND, not merely fail gracefully when absent. On
# 2026-08-17 the edge node had neither ffmpeg nor Homebrew, so every clip
# encoded to nothing — and because the fail-open path worked exactly as
# designed, nothing errored and nothing alerted. Silence is the failure mode
# this asserts against.
print("\nEncoder resolution")
import os                                                        # noqa: E402
import config                                                    # noqa: E402

saved_env = os.environ.pop("FFMPEG_BINARY", None)
real_which = shutil.which
try:
    check("an explicit FFMPEG_BINARY always wins",
          (os.environ.__setitem__("FFMPEG_BINARY", "/explicit/ffmpeg"),
           config._resolve_ffmpeg())[1] == "/explicit/ffmpeg")
    os.environ.pop("FFMPEG_BINARY", None)

    shutil.which = lambda name, *a, **k: None if name == "ffmpeg" else real_which(name, *a, **k)
    config.shutil = shutil
    bundled = config._resolve_ffmpeg()
    check("a host with no ffmpeg still resolves a real encoder",
          bundled != "ffmpeg" and Path(bundled).exists(), bundled)
    check("...and that encoder actually runs",
          subprocess.run([bundled, "-version"], capture_output=True).returncode == 0, bundled)
finally:
    shutil.which = real_which
    config.shutil = shutil
    os.environ.pop("FFMPEG_BINARY", None)
    if saved_env is not None:
        os.environ["FFMPEG_BINARY"] = saved_env

# A deadlock cannot be provoked reliably in a test — it needs ffmpeg to fill a
# 64 KB pipe while still running — so assert the structural property instead:
# nothing on this path may hold a pipe the thread does not drain concurrently.
print("\nNo undrained pipe on the detector thread")
import inspect                                                   # noqa: E402

src = inspect.getsource(det_mod._encode_mp4)
check("stderr is not an undrained PIPE",
      "stderr=subprocess.PIPE" not in src.replace(" ", ""),
      "writes all frames to stdin before reading stderr — that is the deadlock")
check("the encode is bounded by a timeout", "timeout=" in src, src[-400:])

print("\nTrack boxes across observation gaps")
track = det_mod._TrackState(7, "test", 10)
for fi, center in [(10, (20, 20)), (11, (22, 20)), (50, (70, 20))]:
    track.update(center, 0.9, (center[0] - 5, 15, 10, 10), fi)

def rendered(fi, draw_bbox=True):
    canvas = np.zeros((80, 100, 3), dtype=np.uint8)
    count = sum(index <= fi for index in track.frame_indices)
    det_mod._draw_track_on_canvas(canvas, track, (0, 255, 0), fi, count, 1, draw_bbox)
    return canvas

check("a lost track keeps its trail without inventing a box",
      np.array_equal(rendered(30), rendered(30, False)))
check("observed frame retains its box",
      not np.array_equal(rendered(11), rendered(11, False)))
check("nearby frame retains its box",
      not np.array_equal(rendered(12), rendered(12, False)))
check("reappearing track gets its current box",
      not np.array_equal(rendered(50), rendered(50, False)))
print("\nAll trail video checks passed.")
