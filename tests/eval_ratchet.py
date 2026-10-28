"""Model regression gate (棘輪) — evaluate best.pt on the dataset's test split.

ml/datasets/ holds the current dataset export (*.ndjson, from the Ultralytics
platform; the user replaces it as the dataset grows). This script:

  1. reads every `split: "test"` image (including negatives with no boxes),
     from the export AND from tests/extra_test_cases.ndjson (repo-local hard
     cases, which survive the export being replaced),
  2. downloads them into tests/.dataset_cache/ (keyed by the CDN content hash,
     so re-exports with fresh signed URLs reuse the cache); entries with a
     `local` path are read straight from the repo,
  3. runs the model at the deployment settings (imgsz 1600, conf 0.5),
  4. greedy-matches predictions to ground truth at IoU >= 0.5,
  5. reports precision / recall / F1 / mean matched IoU,
  6. checks per-box `must_sight` assertions at the looser --sight-iou — a lost
     capability is worth less F1 than --tolerance permits, so these fail the
     gate on their own,
  7. compares F1 against the baseline for this split (tests/f1_baseline.json for
     test, tests/f1_baseline_<split>.json otherwise) — the RATCHET: fail (exit 1)
     if F1 drops more than --tolerance below it. Pass otherwise;
     --update-baseline records the new score when it improves, and re-records
     unconditionally when the measurement basis changed.

Usage (from repo root):
  uv run --project backend python tests/eval_ratchet.py                # gate
  uv run --project backend python tests/eval_ratchet.py --update-baseline
  uv run --project backend python tests/eval_ratchet.py --model /path/to/new.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXTRA_NDJSON = REPO / "tests" / "extra_test_cases.ndjson"


def find_ndjson(baseline: Path | None = None) -> Path:
    """Which export to score when --ndjson was not given.

    This used to return the alphabetically LAST file with a warning, which made
    the default an accident of naming: committing the four newer benchmark
    exports would have moved it from `monkey-cocotree-july` to
    `monkey-cocotree-sites` (40 images) and a bare run would have started
    scoring a different exam without anyone asking.

    The baseline already records the dataset it was measured on, so it is the
    only defensible default — the file whose numbers this run will be compared
    against. With no baseline and more than one export, there is no right guess,
    so ask rather than pick.
    """
    folder = REPO / "ml" / "datasets"
    candidates = sorted(folder.glob("*.ndjson")) or sorted(REPO.glob("*.ndjson"))
    if not candidates:
        sys.exit("no *.ndjson dataset export found in ml/datasets/")

    if baseline is not None and baseline.exists():
        recorded = json.loads(baseline.read_text()).get("dataset")
        if recorded:
            hit = next((c for c in candidates if c.name == recorded), None)
            if hit is None:
                sys.exit(
                    f"baseline {baseline.name} was measured on {recorded!r}, which is not "
                    f"in {folder}/ — re-export it, or pass --ndjson explicitly."
                )
            return hit

    if len(candidates) > 1:
        sys.exit(
            "multiple dataset exports and no baseline to name one:\n  "
            + "\n  ".join(c.name for c in candidates)
            + "\nPass --ndjson to say which benchmark to score."
        )
    return candidates[0]


def _read_split(ndjson: Path, split: str) -> tuple[str | None, list[dict], list[dict]]:
    name, images, assertions = None, [], []
    for line in ndjson.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") == "dataset":
            name = d.get("name", name)
        elif d.get("type") == "image" and d.get("split") == split:
            images.append(d)
        elif d.get("type") == "assertion" and d.get("split") == split:
            assertions.append(d)
    return name, images, assertions


def _applies(entry: dict, ndjson: Path) -> bool:
    """Is this repo-local case/assertion meant for the export being scored?

    Every entry must say so via `applies_to` (a list of export stems). Absence
    is an error rather than "applies everywhere": there are six benchmarks now,
    and a case written for one of them is by default wrong for the other five.
    """
    scope = entry.get("applies_to")
    if not scope:
        sys.exit(
            f"{EXTRA_NDJSON.name}: entry {entry.get('file', entry.get('match'))!r} has no "
            "'applies_to' — name the export stem(s) it belongs to, e.g. "
            '"applies_to": ["monkey-cocotree-july"]. A case written for one benchmark '
            "must not silently apply to the others."
        )
    return ndjson.stem in scope


BENCHMARKS = ("july", "crowd0813", "sites", "night0814", "neg", "b08140816", "fp0817")


def _origin(entry: dict) -> str:
    """Which benchmark a merged-dataset image came from.

    Six datasets became ONE on 2026-08-17 so the Platform has a single source of
    truth with train/val/test. 量測紀律 still says every model is scored on all
    benchmarks and reported SEPARATELY, never averaged — so the origin travels in
    the filename as a `<tag>__` prefix and `--benchmark` selects it back out. A
    merged score is not a benchmark; it is the average this project refuses.
    """
    return str(entry.get("file", "")).split("__", 1)[0]


def load_test_split(ndjson: Path, split: str = "test",
                    benchmark: str | None = None) -> tuple[str, list[dict]]:
    """Platform export + the repo-local extra cases in EXTRA_NDJSON.

    The export in ml/datasets/ is replaced wholesale on every re-export, so a
    regression case added there would vanish the next time the dataset grows.
    Cases we never want to lose live in tests/extra_test_cases.ndjson instead
    and are merged in here.
    """
    name, images, _ = _read_split(ndjson, split)
    name = name or ndjson.stem

    if EXTRA_NDJSON.exists():
        _, extra, assertions = _read_split(EXTRA_NDJSON, split)
        extra = [e for e in extra if _applies(e, ndjson)]
        assertions = [a for a in assertions if _applies(a, ndjson)]
        if extra:
            images += extra
            print(f"  + {len(extra)} repo-local extra case(s) from {EXTRA_NDJSON.name}")
        # Assertions attach a must_sight guard to an image the export already
        # carries — the image itself lives on the Platform, so merging a copy
        # here would double-count it. An assertion that matches nothing is a
        # hard error: a deleted image must not silently disable its own guard.
        # It is only checked against the export it was WRITTEN for (see
        # _applies): the July assertion hard-exited on all five other
        # benchmarks, where that image was never meant to appear, which is why
        # only one of the six could be scored at all.
        for a in assertions:
            want = a.get("match") or {}
            hits = [im for im in images
                    if all(str(im.get(k)) == str(v) for k, v in want.items())]
            if len(hits) != 1:
                sys.exit(
                    f"assertion for {a.get('file', want)} matched {len(hits)} {split} "
                    "images, expected exactly 1 — deleted from the dataset, or the "
                    "match is ambiguous? Refusing to gate without it."
                )
            hits[0]["must_sight"] = a["must_sight"]
            print(f"  + must_sight on {a.get('file')} (from {EXTRA_NDJSON.name})")

    if benchmark:
        picked = [im for im in images if _origin(im) == benchmark]
        if not picked:
            seen = sorted({_origin(im) for im in images})
            sys.exit(
                f"--benchmark {benchmark!r} matched no {split} image. "
                f"This export carries: {', '.join(seen) or '(no origin prefixes)'}"
            )
        print(f"  benchmark {benchmark}: {len(picked)}/{len(images)} {split} images")
        images = picked

    if not images:
        sys.exit(f"dataset has no {split} split")
    return name, images


def fetch(entry: dict, cache: Path) -> Path | None:
    """Resolve an entry to a local file.

    `local` (repo-relative) wins and needs no network — that is how the
    repo-local extra cases avoid both downloads and signed-URL expiry.
    Otherwise download by CDN content hash; cached files survive re-signing.
    """
    if entry.get("local"):
        path = REPO / entry["local"]
        if path.exists():
            return path
        print(f"  missing local fixture: {entry['local']}")
        return None

    url = entry["url"]
    key = url.split("?")[0].rsplit("/", 1)[-1]
    dst = cache / key
    if dst.exists():
        return dst
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            dst.write_bytes(r.read())
        return dst
    except Exception as e:
        print(f"  download failed for {entry['file']}: {e!r}")
        return None


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _gray_world(img):
    """Scale each channel to the mean of the three. No-op on a balanced frame."""
    import numpy as _np
    f = img.astype(_np.float32)
    means = [f[:, :, i].mean() for i in range(3)]
    target = sum(means) / 3
    for i in range(3):
        f[:, :, i] *= target / max(1e-6, means[i])
    return _np.clip(f, 0, 255).astype(_np.uint8)


def _imgsz_for(args, w: int, h: int) -> int:
    """Per-image inference size. Fixed unless --native, which floors instead."""
    if not args.native:
        return args.imgsz
    return max(args.imgsz, -(-max(int(w), int(h)) // 32) * 32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ndjson", type=Path, default=None)
    p.add_argument("--split", default="test", help="dataset split to evaluate (test|val|train)")
    p.add_argument("--benchmark", default=None,
                   help="score ONE benchmark out of the merged split, by its filename "
                        "origin prefix (%s). Scores are reported per benchmark and "
                        "never averaged." % ", ".join(BENCHMARKS))
    p.add_argument("--model", type=Path, default=REPO / "backend" / "best.pt")
    # Must track config.INFERENCE_IMGSZ (1600). It was left at 1920 after
    # deployment moved to 1600, so every default run scored the model at a
    # resolution the baseline was never recorded at — the comparison below
    # now refuses to proceed on that mismatch rather than reporting a
    # meaningless delta.
    p.add_argument("--imgsz", type=int, default=1600)
    # What production actually does since 2026-08-14: --imgsz is a FLOOR and each
    # image is inferred at its own long side (rounded to 32) when that is larger.
    # A fixed size is what let a 1920x1080 camera be silently squeezed into 1600,
    # and a gate that cannot express the deployed rule cannot police it. The
    # recorded basis becomes "native>=<floor>" so a native run and a fixed run
    # can never be compared to each other.
    p.add_argument("--native", action="store_true",
                   help="never downscale: imgsz = max(--imgsz, ceil32(long side)) per image")
    # Gray-world white balance before inference. The model is far more sensitive
    # to colour cast than the eye is: at dawn the camera's white balance fails
    # (B/R 1.92 vs a normal 1.07) and champion finds 1 box in 59 frames where
    # 0813-fp finds 168. Rebalancing recovers it — and it also helps frames that
    # look fine, +21% recall on the sub-1.15-cast half of the crowd set. A
    # brightness-only control changes nothing, so it is the channel balance and
    # not the exposure.
    p.add_argument("--gray-world", action="store_true",
                   help="normalise per-channel means before inference")
    p.add_argument("--gray-world-above", type=float, default=0.0,
                   help="only correct frames whose max/min channel-mean ratio "
                        "exceeds this (0 = correct everything)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument("--sight-iou", type=float, default=0.25,
                   help="IoU at which a must_sight box counts as seen (looser "
                        "than --iou-match on purpose: presence, not precision)")
    # Split-aware by default. CI gates BOTH --split test and --split val against
    # this flag; with one shared file the val run was being compared against the
    # test split's F1 — different exam, meaningless delta. test keeps the
    # historical filename so nothing external moves.
    p.add_argument("--baseline", type=Path, default=None,
                   help="default: tests/f1_baseline.json for the test split, "
                        "tests/f1_baseline_<split>.json otherwise")
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="allowed F1 drop below baseline before the gate fails")
    p.add_argument("--update-baseline", action="store_true",
                   help="ratchet the baseline UP after an improvement on the same basis")
    p.add_argument("--rebase-basis", action="store_true",
                   help="re-record the baseline when the measurement basis itself changed "
                        "(resolution rule, conf, dataset, split, fp16, image count). Separate "
                        "from --update-baseline on purpose: one raises the bar, this one moves "
                        "the goalposts, and CI must never do the second by accident")
    p.add_argument("--device", default=None, help="torch device (e.g. mps); default auto/cpu")
    p.add_argument("--half", action="store_true",
                   help="force fp16 on the backend (what the edge node runs on MPS)")
    p.add_argument("--tiled", action="store_true",
                   help="evaluate through the ANE tiling path (services/tiling.py) instead of full-frame predict")
    p.add_argument("--fused", action="store_true",
                   help="evaluate the ANE fusion path (full-frame + gated tile sweep)")
    p.add_argument("--tile-model", type=Path, default=None,
                   help="tile model for --fused (defaults to --model, i.e. same weights)")
    args = p.parse_args()
    if args.baseline is None:
        args.baseline = REPO / "tests" / (
            "f1_baseline.json" if args.split == "test" else f"f1_baseline_{args.split}.json"
        )

    ndjson = args.ndjson or find_ndjson(args.baseline)
    ds_name, images = load_test_split(ndjson, args.split, args.benchmark)
    cache = REPO / "tests" / ".dataset_cache"
    cache.mkdir(exist_ok=True)
    print(f"dataset: {ds_name} ({ndjson.name}) — {len(images)} {args.split} images")

    from ultralytics import YOLO
    model = YOLO(str(args.model), task="detect")
    if args.half:
        import numpy as _np
        sys.path.insert(0, str(REPO / "backend"))
        from services.detector import _force_half
        model.predict(_np.zeros((640, 640, 3), dtype=_np.uint8), verbose=False, device=args.device)
        _force_half(model)
    tile_model = None
    if args.tiled or args.fused:
        sys.path.insert(0, str(REPO / "backend"))
        import cv2
        from services.tiling import fused_predict, tiled_predict
        from config import ANE_FUSION_ADD_CONF, ANE_FUSION_GATE_CONF
        if args.fused:
            tile_model = YOLO(str(args.tile_model or args.model), task="detect")

    tp = fp = fn = 0
    matched_ious: list[float] = []
    sightings: list[tuple[str, int, float]] = []
    skipped = 0
    for i, entry in enumerate(images):
        path = fetch(entry, cache)
        if path is None:
            skipped += 1
            continue
        w, h = entry["width"], entry["height"]
        gt = []
        for _, cx, cy, bw, bh in (entry.get("annotations") or {}).get("boxes", []):
            gt.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))

        preds = []
        if args.fused:
            frame = cv2.imread(str(path))
            dets = fused_predict(model, tile_model, frame, conf=args.conf,
                                 imgsz=_imgsz_for(args, w, h), device=args.device,
                                 gate_conf=ANE_FUSION_GATE_CONF, add_conf=ANE_FUSION_ADD_CONF)
            preds = [tuple(map(float, d[:4])) for d in sorted(dets, key=lambda d: -d[4])]
        elif args.tiled:
            frame = cv2.imread(str(path))
            dets = tiled_predict(model, frame, conf=args.conf)
            preds = [tuple(map(float, d[:4])) for d in sorted(dets, key=lambda d: -d[4])]
        else:
            src = str(path)
            if args.gray_world:
                import cv2 as _cv2
                import numpy as _np
                _img = _cv2.imread(str(path))
                _m = [_img[:, :, i].mean() for i in range(3)]
                if max(_m) / max(1e-6, min(_m)) > args.gray_world_above:
                    _img = _gray_world(_img)
                src = _img
            r = model.predict(src, imgsz=_imgsz_for(args, w, h), conf=args.conf,
                              device=args.device, verbose=False, classes=[0])
            boxes = r[0].boxes
            if boxes is not None and len(boxes):
                order = boxes.conf.argsort(descending=True)
                preds = [tuple(boxes.xyxy[j].tolist()) for j in order]

        taken = set()
        for pred in preds:  # greedy: highest-confidence prediction claims first
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gt):
                if j in taken:
                    continue
                v = iou(pred, g)
                if v > best_iou:
                    best_j, best_iou = j, v
            if best_j >= 0 and best_iou >= args.iou_match:
                taken.add(best_j)
                matched_ious.append(best_iou)
                tp += 1
            else:
                fp += 1
        fn += len(gt) - len(taken)

        # Sighting assertions. Aggregate F1 cannot police a single capability:
        # one image among 91 moves F1 by less than the 0.01 tolerance, so the
        # model that stopped seeing the wall climber entirely still PASSED the
        # ratchet. `must_sight` asserts per box that SOME prediction overlaps it
        # at >= --sight-iou — deliberately looser than the 0.5 match IoU,
        # because the question here is "did the detector see the animal at all",
        # not "did it box it precisely". Measured spread on the climber:
        # 0.427 (sees it) vs 0.023 (blind).
        for idx in entry.get("must_sight", []):
            if idx >= len(gt):
                sys.exit(f"{entry['file']}: must_sight index {idx} exceeds {len(gt)} boxes")
            best = max((iou(pred, gt[idx]) for pred in preds), default=0.0)
            sightings.append((entry["file"], idx, best))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(images)} …")

    if skipped:
        print(f"warning: {skipped} images skipped (expired signed URLs? re-export the ndjson)")
        if skipped > len(images) * 0.1:
            sys.exit("more than 10% of the test set unavailable — refusing to gate on a partial set")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    print(f"\nmodel: {args.model.name} @ imgsz {args.imgsz}, conf {args.conf}")
    print(f"TP {tp}  FP {fp}  FN {fn}")
    print(f"precision {precision:.4f} | recall {recall:.4f} | F1 {f1:.4f} | mean IoU {mean_iou:.4f}")

    unsighted = [s for s in sightings if s[2] < args.sight_iou]
    if sightings:
        print(f"\nsighting assertions (IoU >= {args.sight_iou}): "
              f"{len(sightings) - len(unsighted)}/{len(sightings)} sighted")
        for name, idx, best in sightings:
            flag = "ok  " if best >= args.sight_iou else "MISS"
            print(f"  {flag} {name} box[{idx}]  best IoU {best:.3f}")

    def finish() -> None:
        """Sighting losses fail the gate on their own — see must_sight above."""
        if unsighted:
            sys.exit(
                f"SIGHTING FAIL: {len(unsighted)} asserted box(es) unsighted — "
                "this model stopped detecting something a previous one detected"
            )

    # The recorded basis names the RULE, not a number, so a --native run and a
    # fixed-size run can never be silently compared against each other.
    basis_imgsz = f"native>={args.imgsz}" if args.native else args.imgsz
    if args.gray_world:
        basis_imgsz = f"{basis_imgsz}+grayworld>{args.gray_world_above:g}"
    result = {
        "f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
        "mean_iou": round(mean_iou, 4), "dataset": ndjson.name, "test_images": len(images),
        "benchmark": args.benchmark,
        "imgsz": basis_imgsz, "conf": args.conf, "half": bool(args.half), "split": args.split,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if not args.baseline.exists():
        args.baseline.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nno baseline — established: F1 {f1:.4f} → {args.baseline}")
        finish()
        return

    base = json.loads(args.baseline.read_text())

    # An F1 delta only means something when both sides were measured the same
    # way. Comparing a 1920 run against a 1600 baseline silently reported a
    # ~0.02 "regression" that was pure resolution difference.
    # `half` counts as much as imgsz: CI gates with --half (what the edge node
    # runs), and a baseline recorded without it is a different exam — fp16 is
    # worth ~0.002 F1 here, enough to flip a borderline verdict.
    mismatch = [
        f"{k}: baseline {base[k]!r} vs this run {v!r}"
        for k, v in (("imgsz", basis_imgsz), ("conf", args.conf), ("dataset", ndjson.name),
                     ("test_images", len(images)), ("half", bool(args.half)),
                     ("split", args.split), ("benchmark", args.benchmark))
        if k in base and base[k] != v
    ]
    if mismatch:
        # --update-baseline deliberately does NOT re-record here. It used to,
        # which meant `model-promote.yml` — which passes it — could silently
        # re-anchor a baseline under a basis nothing had been gated on. Moving
        # the goalposts now needs someone to say so, in a separate flag.
        if not args.rebase_basis:
            sys.exit(
                "REFUSING TO GATE: baseline was recorded under different settings —\n  "
                + "\n  ".join(mismatch)
                + "\nRe-run matching the baseline, or re-record it with --rebase-basis."
            )
        # The basis changed (harder test set, different resolution), so the old
        # F1 is not a number this run can be compared against — re-record it
        # instead of pretending the delta means something. Deliberately NOT
        # gated on f1 > baseline: adding a hard case is expected to lower F1.
        args.baseline.write_text(json.dumps(result, indent=2) + "\n")
        print("\nbaseline RE-RECORDED — measurement basis changed:\n  " + "\n  ".join(mismatch))
        print(f"new reference point: F1 {f1:.4f} (not a ratchet step; nothing was gated)")
        finish()
        return

    delta = f1 - base["f1"]
    print(f"\nbaseline F1 {base['f1']:.4f} ({base.get('updated_at', '?')}) | delta {delta:+.4f}")

    if f1 < base["f1"] - args.tolerance:
        sys.exit(f"RATCHET FAIL: F1 dropped {-delta:.4f} > tolerance {args.tolerance} — do not ship this model")

    if args.update_baseline and f1 > base["f1"]:
        args.baseline.write_text(json.dumps(result, indent=2) + "\n")
        print(f"baseline updated → F1 {f1:.4f}")
    finish()
    print("RATCHET PASS")


if __name__ == "__main__":
    main()
