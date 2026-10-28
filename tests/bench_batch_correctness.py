"""Does batching change the DETECTIONS, not just the speed?

Relevant because batch inference is the obvious lever for batch-annotation
work — but ultralytics only uses the cheap rectangular letterbox when every
image in a batch shares a shape. Mix a 1920x1080 frame with a 1280x720 one and
each gets padded to a full square instead (2.9x the pixels at imgsz 1600), so
the network sees a different input than it would at batch 1. Whether that
shifts boxes or confidences is an empirical question, and for annotation work
the answer matters more than throughput.

Compares batch=1 against batch=N on the pinned test split:
  - per-detection agreement (geometry pairing + confidence delta)
  - detections present in one run but not the other
  - F1 against ground truth for each mode
  - wall-clock per image

  uv run --project backend python tests/bench_batch_correctness.py --batch 16
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import eval_ratchet as er  # noqa: E402


def run(model, paths, batch, imgsz, device, conf):
    """Return {path: [(xyxy, conf), ...]} plus seconds spent."""
    out = {}
    t0 = time.perf_counter()
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i:i + batch]]
        results = model.predict(chunk if batch > 1 else chunk[0], imgsz=imgsz,
                               conf=conf, device=device, verbose=False)
        for path, r in zip(chunk, results):
            boxes = r.boxes
            dets = []
            if boxes is not None and len(boxes):
                order = boxes.conf.argsort(descending=True)
                dets = [(tuple(boxes.xyxy[j].tolist()), float(boxes.conf[j])) for j in order]
            out[path] = dets
    return out, time.perf_counter() - t0


def score(per_image, gt_by_path):
    tp = fp = fn = 0
    for path, dets in per_image.items():
        gt = gt_by_path[path]
        taken = set()
        for box, _ in dets:
            best_j, best = -1, 0.0
            for j, g in enumerate(gt):
                if j in taken:
                    continue
                v = er.iou(box, g)
                if v > best:
                    best_j, best = j, v
            if best_j >= 0 and best >= 0.5:
                taken.add(best_j)
                tp += 1
            else:
                fp += 1
        fn += len(gt) - len(taken)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=REPO / "backend" / "best.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--group-shapes", action="store_true",
                    help="batch only same-shape images together — ultralytics keeps the "
                         "cheap rectangular letterbox then, instead of padding every "
                         "image to a full square")
    args = ap.parse_args()

    ds_name, images = er.load_test_split(er.find_ndjson(), args.split)
    cache = REPO / "tests" / ".dataset_cache"
    cache.mkdir(exist_ok=True)
    paths, gt_by_path, shapes = [], {}, Counter()
    for entry in images:
        p = er.fetch(entry, cache)
        if p is None:
            continue
        paths.append(p)
        w, h = entry["width"], entry["height"]
        shapes[f"{w}x{h}"] += 1
        gt_by_path[str(p)] = [
            ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            for _, cx, cy, bw, bh in (entry.get("annotations") or {}).get("boxes", [])
        ]
    print(f"{ds_name} [{args.split}] {len(paths)} 張 · 解析度 {dict(shapes)}")
    print(f"模型 {args.model.name} · imgsz {args.imgsz} · conf {args.conf} · {args.device}")
    print(f"批次內混合解析度: {'是（ultralytics 退回正方形填充）' if len(shapes) > 1 else '否'}\n")

    from ultralytics import YOLO
    import numpy as np
    model = YOLO(str(args.model), task="detect")
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=args.device)

    single, t1 = run(model, paths, 1, args.imgsz, args.device, args.conf)
    if args.group_shapes:
        from collections import defaultdict
        groups = defaultdict(list)
        for entry, p_ in zip(images, paths):
            groups[(entry["width"], entry["height"])].append(p_)
        multi, tN = {}, 0.0
        for g in groups.values():
            part, dt = run(model, g, args.batch, args.imgsz, args.device, args.conf)
            multi.update(part)
            tN += dt
        print(f"（依解析度分組：{len(groups)} 組，最大一組 {max(len(g) for g in groups.values())} 張）")
    else:
        multi, tN = run(model, paths, args.batch, args.imgsz, args.device, args.conf)

    deltas, only1, onlyN, moved = [], 0, 0, 0
    for path in map(str, paths):
        a, b = single.get(path, []), multi.get(path, [])
        used = set()
        for box_a, conf_a in a:
            best_j, best = -1, 0.0
            for j, (box_b, _) in enumerate(b):
                if j in used:
                    continue
                v = er.iou(box_a, box_b)
                if v > best:
                    best_j, best = j, v
            if best_j >= 0 and best >= 0.5:
                used.add(best_j)
                if best < 0.9:
                    moved += 1
                deltas.append(abs(conf_a - b[best_j][1]))
            else:
                only1 += 1
        onlyN += len(b) - len(used)

    p1, r1, f1_1 = score(single, gt_by_path)
    pN, rN, f1_N = score(multi, gt_by_path)
    n1 = sum(len(v) for v in single.values())
    nN = sum(len(v) for v in multi.values())

    print(f"{'':14} {'偵測數':>7} {'precision':>10} {'recall':>8} {'F1':>8} {'秒/張':>7}")
    print(f"{'batch 1':14} {n1:>7} {p1:>10.4f} {r1:>8.4f} {f1_1:>8.4f} {t1/len(paths):>7.3f}")
    print(f"{'batch ' + str(args.batch):14} {nN:>7} {pN:>10.4f} {rN:>8.4f} {f1_N:>8.4f} {tN/len(paths):>7.3f}")
    print(f"\n一致性：配對 {len(deltas)} 個（{moved} 個框位移到 IoU 0.5-0.9）")
    print(f"  僅 batch1 有 {only1} 個 · 僅 batch{args.batch} 有 {onlyN} 個")
    if deltas:
        print(f"  信心度差異 中位 {statistics.median(deltas):.4f} · 最大 {max(deltas):.4f}")
    print(f"\n速度 {t1/tN:.2f}x · F1 差異 {f1_N - f1_1:+.4f}")


if __name__ == "__main__":
    main()
