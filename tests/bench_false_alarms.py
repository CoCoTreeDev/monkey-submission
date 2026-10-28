"""Operational false-alarm bench: how often would each model page the user?

The pinned test split answers "does it find a monkey it has never seen"
(213 boxes, verified zero overlap with train). It cannot answer "how many
junk alerts do I delete per day", because 26 background stills are not 24
hours of moving foliage, shifting sun and rain.

So: capture real frames from the live cameras ONCE, then replay that same
footage through every candidate. Detections on empty scenes are false alarms,
and the result is reported in the unit the user actually feels — alerts/hour.

  # 1. capture (run on the edge node, cameras are LAN-visible from there)
  uv run --project backend python tests/bench_false_alarms.py --capture 20
  # 2. replay candidates over the identical footage
  uv run --project backend python tests/bench_false_alarms.py \
      --models backend/best.pt ml/models/0803.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FRAMES = REPO / "tests" / ".falsealarm_frames"


def capture(minutes: float) -> None:
    """Grab frames from every registered camera into a replayable corpus."""
    sys.path.insert(0, str(REPO / "backend"))
    import cv2
    import httpx
    from dotenv import load_dotenv
    import os

    load_dotenv(REPO / "backend" / ".env")
    from services.stream import stream_manager

    api = os.getenv("CLOUD_API_URL", "").rstrip("/")
    tok = httpx.post(f"{api}/api/auth/login",
                     json={"code": os.getenv("CLOUD_ACCESS_CODE")}, timeout=30).json()["token"]
    cams = httpx.get(f"{api}/api/cameras", headers={"Authorization": f"Bearer {tok}"},
                     timeout=30).json()
    FRAMES.mkdir(exist_ok=True)
    for c in cams:
        stream_manager.add(c["camera_id"] + "-fa", c["rtsp_url"])
    time.sleep(12)

    # One frame every 2 s per camera: enough to sample lighting/wind states
    # without turning the corpus into 45 near-identical frames per second.
    deadline = time.time() + minutes * 60
    n = 0
    while time.time() < deadline:
        for c in cams:
            s = stream_manager.get(c["camera_id"] + "-fa")
            frame = s.frame if s else None
            if frame is None:
                continue
            name = f"{c['camera_id']}_{int(time.time())}_{n:05d}.jpg".encode("ascii", "ignore").decode()
            cv2.imwrite(str(FRAMES / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        time.sleep(2)
    stream_manager.stop_all()
    print(f"擷取 {n} 幀 → {FRAMES}（涵蓋 {minutes} 分鐘 × {len(cams)} 支攝影機）")


def replay(models: list[Path], conf: float, imgsz: int, device: str) -> None:
    frames = sorted(FRAMES.glob("*.jpg"))
    if not frames:
        sys.exit(f"沒有畫面可重播 — 先在邊緣節點跑 --capture")
    # Wall-clock the corpus spans, so alert counts convert to alerts/hour.
    stamps = sorted(int(f.stem.split("_")[1]) for f in frames if f.stem.split("_")[1].isdigit())
    span_h = max((stamps[-1] - stamps[0]) / 3600, 1e-6) if len(stamps) > 1 else 1e-6
    cams = len({f.stem.split("_")[0] for f in frames})
    print(f"重播 {len(frames)} 幀 · {cams} 支攝影機 · 涵蓋 {span_h * 60:.0f} 分鐘實際時間\n")

    from ultralytics import YOLO
    sys.path.insert(0, str(REPO / "backend"))
    import numpy as np
    from services.detector import _force_half

    print(f"{'模型':<22} {'觸發幀':>7} {'誤報框':>7} {'每小時':>8} {'最高信心':>9}")
    for mp in models:
        model = YOLO(str(mp), task="detect")
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device)
        _force_half(model)
        hits = boxes_total = 0
        top = 0.0
        for f in frames:
            r = model.predict(str(f), imgsz=imgsz, conf=conf, device=device, verbose=False)
            b = r[0].boxes
            if b is not None and len(b):
                hits += 1
                boxes_total += len(b)
                top = max(top, float(b.conf.max()))
        per_hour = boxes_total / span_h / max(cams, 1)
        print(f"{mp.name:<22} {hits:>7} {boxes_total:>7} {per_hour:>8.1f} {top:>9.2f}")
    print("\n每小時 = 每支攝影機每小時的誤報框數（畫面中確無猴子時，全部都是誤報）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=float, metavar="MINUTES",
                    help="capture live frames into the corpus (run on the edge node)")
    ap.add_argument("--models", nargs="+", type=Path,
                    default=[REPO / "backend" / "best.pt"])
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    if args.capture:
        capture(args.capture)
    else:
        replay(args.models, args.conf, args.imgsz, args.device)


if __name__ == "__main__":
    main()
