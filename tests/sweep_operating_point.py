"""Confidence-threshold sweep: find each model's best operating point.

Retraining is not the only knob. The deployed CONFIDENCE_THRESHOLD (0.5) is a
config value, so a model that looks worse at 0.5 may win at 0.35 — and the
client's real cost function is asymmetric (a false alarm costs manual deletion
every day, a missed monkey costs a missed monkey), which F1 does not express.

Reports, per threshold: precision / recall / F1 and F0.5 (precision-weighted),
plus per-image false alarms — the number the client actually feels.

  uv run --project backend python tests/sweep_operating_point.py \
      --model backend/best.pt --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import eval_ratchet as er  # noqa: E402

THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=REPO / "backend" / "best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--half", action="store_true", default=True)
    args = ap.parse_args()

    ndjson = er.find_ndjson()
    ds_name, images = er.load_test_split(ndjson, args.split)
    cache = REPO / "tests" / ".dataset_cache"
    cache.mkdir(exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(args.model), task="detect")
    if args.half:
        import numpy as np
        sys.path.insert(0, str(REPO / "backend"))
        from services.detector import _force_half
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=args.device)
        _force_half(model)

    # One inference pass at the lowest threshold; every higher threshold is a
    # filter on the same predictions, so the sweep costs one pass, not seven.
    per_image = []
    for i, entry in enumerate(images):
        path = er.fetch(entry, cache)
        if path is None:
            continue
        w, h = entry["width"], entry["height"]
        gt = [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
              for _, cx, cy, bw, bh in (entry.get("annotations") or {}).get("boxes", [])]
        r = model.predict(str(path), imgsz=args.imgsz, conf=min(THRESHOLDS),
                          device=args.device, verbose=False)
        boxes = r[0].boxes
        dets = []
        if boxes is not None and len(boxes):
            order = boxes.conf.argsort(descending=True)
            dets = [(tuple(boxes.xyxy[j].tolist()), float(boxes.conf[j])) for j in order]
        per_image.append((gt, dets))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(images)} …", flush=True)

    print(f"\n{args.model.name} @ {ds_name} [{args.split}] — {len(per_image)} 張, imgsz {args.imgsz}")
    print(f"{'conf':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>9} {'recall':>7} "
          f"{'F1':>7} {'F0.5':>7} {'誤報/張':>8}")
    for thr in THRESHOLDS:
        tp = fp = fn = 0
        for gt, dets in per_image:
            taken: set[int] = set()
            for box, conf in dets:
                if conf < thr:
                    continue
                best_j, best_iou = -1, 0.0
                for j, g in enumerate(gt):
                    if j in taken:
                        continue
                    v = er.iou(box, g)
                    if v > best_iou:
                        best_j, best_iou = j, v
                if best_j >= 0 and best_iou >= 0.5:
                    taken.add(best_j)
                    tp += 1
                else:
                    fp += 1
            fn += len(gt) - len(taken)
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        f05 = 1.25 * p * r_ / (0.25 * p + r_) if p + r_ else 0.0
        print(f"{thr:>5} {tp:>4} {fp:>4} {fn:>4} {p:>9.4f} {r_:>7.4f} "
              f"{f1:>7.4f} {f05:>7.4f} {fp / max(len(per_image), 1):>8.2f}")


if __name__ == "__main__":
    main()
