"""Tiled CoreML inference for the Apple Neural Engine (ANE).

The exported CoreML model has a fixed 640x640 input, so full frames are
split into overlapping 640x640 tiles, each tile is inferred independently
(the ANE runs these without contending with the MPS/GPU path), the boxes
are offset back into global frame coordinates, and duplicates in the tile
overlap regions are removed with cross-tile NMS.

Tracking cannot go through model.track() here (that API owns its own
predictor/tracker wiring), so TiledTracker instantiates a BYTETracker
configured exactly like ultralytics' bytetrack.yaml flow and feeds it the
merged global-coordinate detections.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Tile geometry: fixed CoreML input size, minimum overlap between adjacent
# tiles (15% of tile size) so objects on a seam appear whole in at least one
# tile, and the IoU threshold for cross-tile duplicate suppression.
_TILE = 640
_MIN_OVERLAP = 96
_NMS_IOU = 0.55
_IOM_SUPPRESS = 0.6
# Fusion duplicate test — see _is_duplicate_of for why these are not one IoM.
_FUSION_DUP_IOU = 0.5
_FUSION_DUP_CONTAINMENT = 0.75


def compute_tile_origins(width: int, height: int) -> list[tuple[int, int]]:
    """Compute (x, y) tile origins covering a width x height frame.

    Per axis: the minimum number of 640px tiles that covers the span with at
    least _MIN_OVERLAP px of overlap between neighbours, spread evenly from
    edge to edge. For 1920x1080 this yields 4 columns (x=0,427,853,1280) by
    2 rows (y=0,440) = 8 tiles. Spans <= 640 use a single tile at 0 (the
    model letterboxes smaller inputs).
    """

    def axis_origins(span: int) -> list[int]:
        if span <= _TILE:
            return [0]
        n = math.ceil((span - _TILE) / (_TILE - _MIN_OVERLAP)) + 1
        return [round(i * (span - _TILE) / (n - 1)) for i in range(n)]

    return [(x, y) for y in axis_origins(height) for x in axis_origins(width)]


def tiled_predict(model, frame: np.ndarray, conf: float) -> np.ndarray:
    """Run tiled inference on a full frame.

    Crops overlapping 640x640 tiles, predicts each tile, offsets boxes back
    to global frame coordinates, and removes cross-tile duplicates with
    per-class NMS. Returns a float32 (N, 6) array of
    [x1, y1, x2, y2, conf, cls] in global coordinates ((0, 6) when empty).
    """
    h, w = frame.shape[:2]

    boxes_parts: list[np.ndarray] = []
    for x, y in compute_tile_origins(w, h):
        tile = frame[y:y + _TILE, x:x + _TILE]
        results = model.predict(tile, imgsz=_TILE, conf=conf, classes=[0], verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            xyxy[:, [0, 2]] += x
            xyxy[:, [1, 3]] += y
            part = np.concatenate(
                [
                    xyxy,
                    boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1),
                    boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1),
                ],
                axis=1,
            )
            boxes_parts.append(part)

    if not boxes_parts:
        return np.zeros((0, 6), dtype=np.float32)

    dets = np.concatenate(boxes_parts, axis=0)

    # Cross-tile NMS, per class: duplicates of the same object appear in the
    # overlap regions of adjacent tiles. NMSBoxes takes top-left xywh.
    keep_mask = np.zeros(len(dets), dtype=bool)
    xywh_tl = dets[:, :4].copy()
    xywh_tl[:, 2] -= xywh_tl[:, 0]
    xywh_tl[:, 3] -= xywh_tl[:, 1]
    for cls_val in np.unique(dets[:, 5]):
        cls_idx = np.flatnonzero(dets[:, 5] == cls_val)
        kept = cv2.dnn.NMSBoxes(
            xywh_tl[cls_idx].tolist(),
            dets[cls_idx, 4].tolist(),
            conf,
            _NMS_IOU,
        )
        if len(kept) > 0:
            keep_mask[cls_idx[np.asarray(kept).flatten()]] = True

    dets = dets[keep_mask]

    # Second pass: intersection-over-minimum suppression. Partial views of
    # one object from adjacent tiles are clipped differently, so their
    # mutual IoU stays below _NMS_IOU and plain NMS keeps both — but the
    # clipped view is almost fully CONTAINED in the fuller one. Suppress any
    # lower-confidence box whose intersection covers >= _IOM_SUPPRESS of the
    # smaller box's area (measured on the test split: 52 of 60 tiled-path
    # false positives were such duplicates/clipped fragments).
    order = np.argsort(-dets[:, 4])
    keep: list[int] = []
    for i in order:
        suppressed = False
        for j in keep:
            if dets[i, 5] != dets[j, 5]:
                continue
            ix = max(0.0, min(dets[i, 2], dets[j, 2]) - max(dets[i, 0], dets[j, 0]))
            iy = max(0.0, min(dets[i, 3], dets[j, 3]) - max(dets[i, 1], dets[j, 1]))
            inter = ix * iy
            area_i = (dets[i, 2] - dets[i, 0]) * (dets[i, 3] - dets[i, 1])
            area_j = (dets[j, 2] - dets[j, 0]) * (dets[j, 3] - dets[j, 1])
            min_area = min(area_i, area_j)
            if min_area > 0 and inter / min_area >= _IOM_SUPPRESS:
                suppressed = True
                break
        if not suppressed:
            keep.append(int(i))

    return dets[sorted(keep)]


def fused_predict(
    frame_model,
    tile_model,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    gate_conf: float,
    add_conf: float,
) -> np.ndarray:
    """Full-frame inference plus a GATED ANE tile sweep.

    The full-frame pass (GPU) runs every frame and anchors precision. When it
    sees anything at or above `gate_conf` the scene is interesting, so a tile
    sweep (ANE, independent silicon) runs as well and contributes any confident
    detection the full-frame pass missed entirely — typically small/distant
    animals that survive better at native 640 than letterboxed to `imgsz`.

    Empty scenes skip the tile sweep, so the added cost is confined to frames
    that already contain a subject (~1% in production).

    Returns float32 (N, 6) [x1, y1, x2, y2, conf, cls] in frame coordinates.
    """
    results = frame_model.predict(frame, imgsz=imgsz, conf=conf, classes=[0], verbose=False, device=device)
    parts: list[np.ndarray] = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        parts.append(
            np.concatenate(
                [
                    boxes.xyxy.cpu().numpy().astype(np.float32),
                    boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1),
                    boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1),
                ],
                axis=1,
            )
        )
    full = np.concatenate(parts, axis=0) if parts else np.zeros((0, 6), dtype=np.float32)

    if tile_model is None or len(full) == 0 or float(full[:, 4].max()) < gate_conf:
        return full

    tiles = tiled_predict(tile_model, frame, conf=add_conf)
    extras = [t for t in tiles if not any(_is_duplicate_of(t, f) for f in full)]
    if not extras:
        return full
    return np.concatenate([full, np.asarray(extras, dtype=np.float32)], axis=0)


def _is_duplicate_of(tile_box, full_box) -> bool:
    """Does `tile_box` describe an object the full-frame pass already found?

    This used to be `_iom(tile, full) < 0.3`, and intersection-over-MINIMUM is
    the wrong question here: it divides by the smaller box's area, so a tall
    thin box that merely CROSSES a bigger one scores high while describing a
    different animal. Measured on the 5-macaque regression image: the tile
    sweep's box for the wall climber, [570,659,640,971], crosses the road
    monkey's [557,829,708,942] with IoM 0.46 — over the old 0.3 cutoff, so the
    only detection of that animal was thrown away. Its real IoU is 0.26 and
    only 36% of it lies inside the other box.

    Two things actually mean "duplicate":
      - high IoU — the two boxes agree on one object;
      - or the tile box is almost entirely INSIDE the full box, which is the
        clipped-partial-view case IoM was reaching for. That needs a high bar
        (0.75), not 0.3: a genuine clipped view sits ~fully inside the fuller
        box, whereas a crossing box only overlaps a slice of itself.

    Note `tiled_predict`'s own cross-tile `_IOM_SUPPRESS` pass has the same
    shape but a different job (both boxes there are tile-clipped views of the
    same frame, and its 0.6 threshold was measured against real duplicates), so
    it is deliberately left alone.
    """
    return _iou(tile_box, full_box) >= _FUSION_DUP_IOU or (
        _containment(tile_box, full_box) >= _FUSION_DUP_CONTAINMENT
    )


def _iou(a, b) -> float:
    ix = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    iy = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(inner, outer) -> float:
    """Fraction of `inner`'s own area that lies inside `outer`."""
    ix = max(0.0, min(float(inner[2]), float(outer[2])) - max(float(inner[0]), float(outer[0])))
    iy = max(0.0, min(float(inner[3]), float(outer[3])) - max(float(inner[1]), float(outer[1])))
    inter = ix * iy
    area = (float(inner[2]) - float(inner[0])) * (float(inner[3]) - float(inner[1]))
    return inter / area if area > 0 else 0.0


class _Detections:
    """Minimal stand-in for the Boxes object BYTETracker.update() consumes.

    BYTETracker only touches .conf, .cls, .xywh (CENTER-x, center-y, w, h),
    len(), and boolean-mask indexing. Deliberately does NOT define .xywhr
    (its presence would flip the tracker into OBB mode) nor .xyxy (only read
    on the BOTSORT gmc branch, unreachable for BYTETracker).
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_Detections":
        return _Detections(self.xywh[mask], self.conf[mask], self.cls[mask])


class TiledTracker:
    """BYTETracker wrapper for the tiled path.

    Mirrors ultralytics' on_predict_start construction exactly (same tracker
    yaml, frame_rate=30) so tracking behaviour matches model.track(). It has to
    read the SAME file the standard path passes to model.track() — when this
    one still loaded stock bytetrack.yaml, raising track_buffer fixed the
    cameras on the standard path and silently left the ANE ones fragmenting.
    """

    def __init__(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        from config import TRACKER_YAML

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKER_YAML)))
        self._tracker = BYTETracker(args=cfg, frame_rate=30)

    def update(self, dets: np.ndarray) -> np.ndarray:
        """Feed (N, 6) [x1, y1, x2, y2, conf, cls] global detections.

        Returns the tracker's float32 array of activated tracks with columns
        [x1, y1, x2, y2, track_id, score, cls, idx] (Kalman-smoothed boxes,
        same as model.track() results) — may be empty (shape (0,)).
        """
        if len(dets) == 0:
            xywh = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        else:
            x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
            xywh = np.stack(
                [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1,
            ).astype(np.float32)
            conf = dets[:, 4].astype(np.float32)
            cls = dets[:, 5].astype(np.float32)
        # img/feats omitted: BYTETracker never sets .gmc, so the img branch is
        # unreachable (BOTSORT-only) and feats are unused.
        return self._tracker.update(_Detections(xywh, conf, cls))


class StandardTracker:
    """External ByteTrack state for a batched standard-model prediction.

    ``model.track`` owns one tracker inside its predictor. That is correct for
    one camera, but a shared predictor would mix cameras. The POC batcher uses
    ``model.predict`` for the shared forward pass and gives each camera one of
    these private trackers. The update is the same operation as Ultralytics'
    ``on_predict_postprocess_end`` callback.
    """

    def __init__(self) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        from config import TRACKER_YAML

        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKER_YAML)))
        self._tracker = BYTETracker(args=cfg, frame_rate=30)

    def update(self, result) -> np.ndarray:
        """Return Ultralytics-compatible rows for one raw Results object."""
        boxes = result.boxes
        if boxes is None:
            # Keep ByteTrack's frame clock moving even for a result without a
            # Boxes object. DetectionPredictor normally supplies an empty
            # Boxes instance, but this preserves model.track semantics for
            # alternate detect backends too.
            detections = _Detections(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        else:
            # This is deliberately the same conversion used by
            # ultralytics.trackers.track.on_predict_postprocess_end.
            detections = boxes.cpu().numpy()
        tracks = self._tracker.update(detections, result.orig_img)
        if len(tracks) == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return tracks
