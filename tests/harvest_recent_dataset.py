"""Harvest recent production detection trails into a labelled YOLO upload.

The production API stores one raw-frame archive per trail.  Several detection
rows can point at the same trail when multiple macaques were visible together,
so this groups by trail before sampling; otherwise one visit would be uploaded
four times with almost identical frames.

This script deliberately only prepares a local YOLO directory.  The existing
``tests/fetch_platform.py --upload-labeled`` command performs the verified
Platform upload after the output has been inspected.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile

import cv2
import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://localhost:8000"
FRAME_RE = re.compile(r"raw[_-]?frame(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Frame:
    index: int
    source_name: str
    data: bytes
    dhash: int


@dataclass(frozen=True)
class Snapshot:
    data: bytes
    dhash: int
    width: int
    height: int


def dhash(data: bytes) -> int | None:
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    small = cv2.resize(image, (9, 8))
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(small[row, col + 1] > small[row, col])
    return bits


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def default_api_url() -> str:
    env = REPO / "frontend" / ".env.local"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("NEXT_PUBLIC_API_URL="):
                return line.split("=", 1)[1].strip().rstrip("/")
    return DEFAULT_API_URL


def login(client: httpx.Client, api_url: str, code: str) -> str:
    response = client.post(f"{api_url}/api/auth/login", json={"code": code})
    response.raise_for_status()
    token = response.json().get("token")
    if not token:
        raise RuntimeError("API login response did not contain a token")
    return str(token)


def fetch_records(
    client: httpx.Client, api_url: str, token: str, start: str, end: str
) -> list[dict]:
    """Read the paginated detection list and retain only Taiwan-local dates."""
    headers = {"Authorization": f"Bearer {token}"}
    records: dict[int, dict] = {}
    offset = 0
    while True:
        response = client.get(
            f"{api_url}/api/detections",
            headers=headers,
            params={"limit": 200, "offset": offset, "sort_by": "time", "sort_order": "desc"},
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items") or []
        if not items:
            break
        for item in items:
            day = str(item.get("timestamp") or "")[:10]
            if start <= day <= end:
                records[int(item["id"])] = item
        offset += len(items)
        if offset >= int(payload.get("total") or 0):
            break
    return sorted(records.values(), key=lambda r: (str(r.get("timestamp")), int(r["id"])))


def download_zip(
    client: httpx.Client, api_url: str, token: str, detection_id: int
) -> list[Frame]:
    response = client.get(
        f"{api_url}/api/detections/{detection_id}/frames",
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()
    frames: list[Frame] = []
    with ZipFile(io.BytesIO(response.content)) as archive:
        names = sorted(
            archive.namelist(),
            key=lambda name: int(FRAME_RE.search(name).group(1))
            if FRAME_RE.search(name)
            else 10**9,
        )
        for fallback_index, name in enumerate(names):
            data = archive.read(name)
            image_hash = dhash(data)
            if image_hash is None:
                continue
            match = FRAME_RE.search(name)
            index = int(match.group(1)) if match else fallback_index
            frames.append(Frame(index, name, data, image_hash))
    if not frames:
        raise RuntimeError(f"detection {detection_id}: raw archive contained no decodable JPGs")
    return frames


def download_snapshot(
    client: httpx.Client, api_url: str, token: str, filename: str
) -> Snapshot | None:
    if not filename:
        return None
    response = client.get(
        f"{api_url}/api/snapshots/{quote(filename, safe='')}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if response.is_error:
        return None
    image_hash = dhash(response.content)
    image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_hash is None or image is None:
        return None
    return Snapshot(response.content, image_hash, image.shape[1], image.shape[0])


def _distance_to(hashes: list[int], value: int, default: int = 64) -> int:
    return min((hamming(value, other) for other in hashes), default=default)


def select_diverse(
    frames: list[Frame],
    anchor_hashes: list[int],
    count: int,
    global_hashes: list[int],
) -> list[Frame]:
    """Select a fixed quota with temporal spread and dHash diversity.

    A detection snapshot anchors the first choice.  Later choices maximise
    distance from both this trail and previously selected trails, while a
    temporal term prevents all choices clustering at one end of the archive.
    The threshold is a preference rather than a hard quota: each trail still
    contributes ``count`` frames when its footage is visually repetitive.
    """
    if not frames or count <= 0:
        return []
    frames = sorted(frames, key=lambda frame: frame.index)
    selected: list[Frame] = []
    remaining = list(frames)
    span = max(1, frames[-1].index - frames[0].index)

    if anchor_hashes:
        first = min(
            remaining,
            key=lambda frame: (
                _distance_to(anchor_hashes, frame.dhash),
                _distance_to(global_hashes, frame.dhash),
                frame.index,
            ),
        )
    else:
        first = remaining[len(remaining) // 2]
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < count:
        def score(frame: Frame) -> tuple[float, float, int]:
            own_distance = _distance_to([item.dhash for item in selected], frame.dhash)
            global_distance = _distance_to(global_hashes, frame.dhash)
            time_distance = min(
                abs(frame.index - item.index) / span for item in selected
            )
            anchor_distance = _distance_to(anchor_hashes, frame.dhash)
            # Visual novelty dominates. Temporal spread breaks ties and the
            # anchor term keeps at least some frames near the recorded sighting.
            value = own_distance * 3.0 + global_distance * 1.5 + time_distance * 12.0
            value -= anchor_distance * 0.15
            return value, time_distance, -frame.index

        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return sorted(selected, key=lambda frame: frame.index)


def write_contact_sheet(output: Path, selected: list[tuple[str, Frame]]) -> None:
    thumbs: list[np.ndarray] = []
    for label, frame in selected:
        image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (320, 180))
        cv2.rectangle(image, (0, 0), (319, 24), (25, 25, 25), -1)
        cv2.putText(image, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(image)
    if not thumbs:
        return
    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    blank = np.zeros_like(thumbs[0])
    sheet = np.vstack([
        np.hstack(thumbs[row * columns:(row + 1) * columns]
                  + [blank] * (columns - len(thumbs[row * columns:(row + 1) * columns])))
        for row in range(rows)
    ])
    cv2.imwrite(str(output), sheet)


def prelabel(
    output: Path,
    model_path: Path,
    device: str,
    imgsz: int,
    conf: float,
    known_labels: dict[str, list[list[float]]] | None = None,
) -> dict:
    sys.path.insert(0, str(REPO / "backend"))
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    image_paths = sorted((output / "images").iterdir())
    result_rows: dict[str, dict] = {}
    # Match config.imgsz_for_frame(): the floor is 1600, but a larger source
    # frame is never squeezed down.  Ultralytics accepts one imgsz per call,
    # so process each native long-side bucket separately.
    by_size: dict[int, list[Path]] = {}
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot decode selected image {path}")
        long_side = max(image.shape[:2])
        native_size = max(imgsz, ((long_side + 31) // 32) * 32)
        by_size.setdefault(native_size, []).append(path)

    for infer_size, paths in sorted(by_size.items()):
        print(f"  預標記 imgsz={infer_size}：{len(paths)} 張")
        results = model.predict(
            source=[str(path) for path in paths],
            imgsz=infer_size,
            conf=conf,
            device=device,
            verbose=False,
            stream=True,
        )
        for path, result in zip(paths, results):
            rows: list[list[float]] = []
            confidences: list[float] = []
            if result.boxes is not None:
                xywhn = result.boxes.xywhn.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                for box, cls in zip(xywhn, classes):
                    if int(cls) != 0:
                        continue
                    rows.append([0.0, *[round(float(value), 6) for value in box]])
            # The production row already carries the detector's box for its
            # best frame. Keep that box on the exact snapshot anchor even if
            # the local candidate model misses this very small macaque.
            if known_labels and path.name in known_labels:
                rows = known_labels[path.name]
            label_path = output / "labels" / f"{path.stem}.txt"
            label_path.write_text(
                "\n".join(" ".join(f"{value:.6f}" for value in row) for row in rows)
                + ("\n" if rows else "")
            )
            result_rows[path.name] = {
                "boxes": len(rows),
                "max_confidence": round(max(confidences), 4) if confidences else None,
                "imgsz": infer_size,
                "source": "production_detection_box" if known_labels and path.name in known_labels else "local_model",
            }
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-08-31")
    parser.add_argument("--end", default="2026-09-01")
    parser.add_argument("--api-url", default=default_api_url())
    parser.add_argument("--access-code", default=os.getenv("MONKEY_ACCESS_CODE"))
    parser.add_argument("--output", type=Path,
                        default=Path("/private/tmp/monkey-harvest-20260831-0901"))
    parser.add_argument("--per-trail", type=int, default=3,
                        help="frames per unique trail archive")
    parser.add_argument("--model", type=Path, default=REPO / "backend" / "best.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if not args.access_code:
        raise SystemExit("請用 --access-code 或 MONKEY_ACCESS_CODE 提供正式 API access code")
    if not args.model.exists():
        raise SystemExit(f"找不到模型：{args.model}")

    output = args.output.resolve()
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0), follow_redirects=True) as client:
        token = login(client, args.api_url.rstrip("/"), args.access_code)
        records = fetch_records(client, args.api_url.rstrip("/"), token, args.start, args.end)
        if not records:
            raise SystemExit(f"{args.start}～{args.end} 找不到偵測紀錄")

        groups: dict[str, list[dict]] = {}
        for record in records:
            key = str(record.get("trail_path") or f"detection-{record['id']}")
            groups.setdefault(key, []).append(record)
        print(f"偵測紀錄：{len(records)} 筆；共用 trail 後：{len(groups)} 組")

        manifest: dict = {
            "start": args.start,
            "end": args.end,
            "records": records,
            "groups": [],
            "selection": [],
        }
        selected_for_sheet: list[tuple[str, Frame]] = []
        global_hashes: list[int] = []
        known_labels: dict[str, list[list[float]]] = {}
        image_counter = 0

        for group_number, (trail, group_records) in enumerate(groups.items(), 1):
            source_id = int(group_records[0]["id"])
            frames = download_zip(client, args.api_url.rstrip("/"), token, source_id)
            snapshots: list[tuple[dict, Snapshot]] = []
            for record in group_records:
                snapshot = download_snapshot(
                    client, args.api_url.rstrip("/"), token,
                    str(record.get("image_path") or ""),
                )
                if snapshot is not None:
                    snapshots.append((record, snapshot))

            # Keep one exact production snapshot per detection row so every
            # record has a known positive anchor. Add diverse trail frames as
            # context; for a shared multi-animal trail this is N anchors + 2
            # supplements, not N copies of the same three frames.
            raw_count = max(1, args.per_trail - 1)
            anchor_hashes = [snapshot.dhash for _, snapshot in snapshots]
            chosen = select_diverse(
                frames, [], raw_count, [*global_hashes, *anchor_hashes]
            )
            group_manifest = {
                "trail_path": trail,
                "record_ids": [int(record["id"]) for record in group_records],
                "source_detection_id": source_id,
                "raw_frame_count": len(frames),
                "selected": [],
            }
            for record, snapshot in snapshots:
                image_counter += 1
                filename = f"record{int(record['id']):05d}_anchor.jpg"
                (images_dir / filename).write_bytes(snapshot.data)
                width = max(1, snapshot.width)
                height = max(1, snapshot.height)
                known_labels[filename] = [[
                    0.0,
                    float(record["bbox_x"]) / width,
                    float(record["bbox_y"]) / height,
                    float(record["bbox_w"]) / width,
                    float(record["bbox_h"]) / height,
                ]]
                group_manifest["selected"].append({
                    "filename": filename,
                    "role": "production_snapshot_anchor",
                    "record_id": int(record["id"]),
                    "dhash": f"{snapshot.dhash:016x}",
                })
                manifest["selection"].append({
                    "filename": filename,
                    "trail_path": trail,
                    "record_ids": [int(record["id"])],
                    "role": "production_snapshot_anchor",
                })
                anchor_frame = Frame(-int(record["id"]), filename, snapshot.data, snapshot.dhash)
                selected_for_sheet.append((filename, anchor_frame))
                global_hashes.append(snapshot.dhash)
            for frame in chosen:
                image_counter += 1
                filename = f"trail{group_number:02d}_frame{frame.index:04d}.jpg"
                (images_dir / filename).write_bytes(frame.data)
                group_manifest["selected"].append({
                    "filename": filename,
                    "frame_index": frame.index,
                    "dhash": f"{frame.dhash:016x}",
                })
                manifest["selection"].append({
                    "filename": filename,
                    "trail_path": trail,
                    "record_ids": group_manifest["record_ids"],
                    "frame_index": frame.index,
                    "role": "diverse_trail_frame",
                })
                selected_for_sheet.append((filename, frame))
                global_hashes.append(frame.dhash)
            manifest["groups"].append(group_manifest)
            distances = [
                hamming(left.dhash, right.dhash)
                for index, left in enumerate(chosen)
                for right in chosen[index + 1:]
            ]
            print(
                f"  組 {group_number}: {len(group_records)} 筆 / {len(frames)} frames "
                f"→ {len(chosen)} 張；組內最小 dHash 距離 "
                f"{min(distances) if distances else '-'}"
            )

    print(f"抽樣完成：{image_counter} 張，開始用 {args.model.name} 預標記…")
    labels = prelabel(output, args.model, args.device, args.imgsz, args.conf, known_labels)
    nonempty = sum(bool(row["boxes"]) for row in labels.values())
    empty = len(labels) - nonempty
    print(f"預標記：{nonempty}/{len(labels)} 張有 monkey box；{empty} 張沒有 box")
    if empty:
        print("WARNING: 沒有 box 的影像仍保留，送入 Platform 前請在 UI 複核，避免誤當負樣本。")

    (output / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: monkey\n"
    )
    manifest["prelabel"] = {
        "model": str(args.model),
        "device": args.device,
        "imgsz_floor": args.imgsz,
        "conf": args.conf,
        "images": labels,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    write_contact_sheet(output / "contact-sheet.jpg", selected_for_sheet)
    print(f"輸出：{output}")
    print("下一步：先檢查 contact-sheet.jpg，再用 fetch_platform.py --upload-labeled 上傳 train split。")


if __name__ == "__main__":
    main()
