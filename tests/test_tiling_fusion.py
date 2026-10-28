"""Regression test for the fusion duplicate filter in services/tiling.py.

No model, no camera, no network — the filter is pure geometry, so it is driven
directly with the box coordinates measured on the 5-macaque regression image.

The bug: `fused_predict` decided whether a tile detection was already covered by
the full-frame pass with intersection-over-MINIMUM. That divides by the smaller
box's area, so a tall thin box that merely CROSSES a bigger one scores high and
gets discarded as a duplicate. On the regression image this silently threw away
the only detection of the wall climber — the exact animal the whole tile sweep
exists to catch.

Run:  uv run --project backend python tests/test_tiling_fusion.py
(also collectable by pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

import numpy as np  # noqa: E402

from services.tiling import _containment, _is_duplicate_of, _iou, fused_predict, tiled_predict  # noqa: E402

# Measured on tests/fixtures/macaque-group-closeup-2026-08-12.jpg.
CLIMBER_TILE = [570.0, 659.0, 640.0, 971.0, 0.155, 0.0]   # tall, thin, real animal
ROAD_FULL = [557.0, 829.0, 708.0, 942.0, 0.908, 0.0]      # different animal, crossed by it


def test_crossing_box_is_not_a_duplicate():
    """The exact geometry that used to be suppressed."""
    assert _iou(CLIMBER_TILE, ROAD_FULL) < 0.5, "these are different animals"
    assert _containment(CLIMBER_TILE, ROAD_FULL) < 0.75, "only a slice of it is inside"
    # the old test: intersection / min(area) — for the record, this is what fired
    inter = (min(CLIMBER_TILE[2], ROAD_FULL[2]) - max(CLIMBER_TILE[0], ROAD_FULL[0])) * (
        min(CLIMBER_TILE[3], ROAD_FULL[3]) - max(CLIMBER_TILE[1], ROAD_FULL[1])
    )
    area_t = (CLIMBER_TILE[2] - CLIMBER_TILE[0]) * (CLIMBER_TILE[3] - CLIMBER_TILE[1])
    area_f = (ROAD_FULL[2] - ROAD_FULL[0]) * (ROAD_FULL[3] - ROAD_FULL[1])
    old_iom = inter / min(area_t, area_f)
    assert old_iom >= 0.3, "the old cutoff really did fire on this pair"

    assert not _is_duplicate_of(CLIMBER_TILE, ROAD_FULL), \
        "a box that merely crosses another must survive"
    print(f"ok crossing box kept (old IoM {old_iom:.2f} would have dropped it, "
          f"IoU {_iou(CLIMBER_TILE, ROAD_FULL):.2f})")


def test_clipped_partial_view_is_still_a_duplicate():
    """The case IoM was reaching for must keep working: a tile shows part of an
    animal the full-frame box already covers whole."""
    full = [100.0, 100.0, 300.0, 300.0, 0.9, 0.0]
    clipped = [110.0, 110.0, 220.0, 290.0, 0.4, 0.0]      # sits inside it
    assert _iou(clipped, full) < 0.5, "a clipped view has low IoU with the fuller box"
    assert _is_duplicate_of(clipped, full), "…but it is still the same animal"
    print("ok clipped partial view still suppressed")


def test_same_object_high_iou_is_a_duplicate():
    a = [100.0, 100.0, 200.0, 200.0, 0.5, 0.0]
    b = [104.0, 103.0, 203.0, 199.0, 0.9, 0.0]
    assert _is_duplicate_of(a, b)
    print("ok near-identical boxes suppressed")


def test_disjoint_box_survives():
    a = [10.0, 10.0, 50.0, 50.0, 0.5, 0.0]
    b = [400.0, 400.0, 500.0, 500.0, 0.9, 0.0]
    assert not _is_duplicate_of(a, b)
    print("ok disjoint box kept")


class _StubModel:
    """Returns a fixed detection set, shaped like an ultralytics result."""

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, *_a, **_k):
        arr = np.asarray(self._boxes, dtype=np.float32).reshape(-1, 6)
        if _k.get("classes") is not None:
            arr = arr[np.isin(arr[:, 5], _k["classes"])]

        class _T:
            def __init__(self, v):
                self._v = v

            def cpu(self):
                return self

            def numpy(self):
                return self._v

        class _B:
            def __init__(self, a):
                self.xyxy = _T(a[:, :4])
                self.conf = _T(a[:, 4])
                self.cls = _T(a[:, 5])

            def __len__(self):
                return len(self.xyxy._v)

        return [type("R", (), {"boxes": _B(arr)})()]


def test_fused_predict_keeps_the_climber(monkeypatch=None):
    """End-to-end through the real fused_predict, with the tile sweep stubbed."""
    import services.tiling as tiling

    frame = np.zeros((1094, 2048, 3), dtype=np.uint8)
    original = tiling.tiled_predict
    tiling.tiled_predict = lambda *_a, **_k: np.asarray([CLIMBER_TILE], dtype=np.float32)
    try:
        out = fused_predict(
            frame_model=_StubModel([ROAD_FULL]),
            tile_model=object(),
            frame=frame,
            conf=0.5,
            imgsz=1600,
            device=None,
            gate_conf=0.5,
            add_conf=0.1,
        )
    finally:
        tiling.tiled_predict = original

    assert len(out) == 2, f"the tile detection must survive, got {len(out)} boxes"
    kept = [row for row in out if abs(row[0] - CLIMBER_TILE[0]) < 1e-3]
    assert kept, "the climber box is missing from the fused output"
    print("ok fused_predict keeps the crossing tile detection (2 boxes out)")


def test_humans_never_enter_tiled_or_fused_tracking():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    monkey = [10, 10, 50, 50, 0.8, 0]
    human = [100, 100, 200, 300, 0.99, 1]
    for rows in ([human], [monkey, human]):
        model = _StubModel(rows)
        for out in (
            tiled_predict(model, frame, 0.1),
            fused_predict(model, model, frame, 0.1, 640, None, 0.5, 0.1),
        ):
            assert len(out) == len(rows) - 1
            assert np.all(out[:, 5] == 0)


if __name__ == "__main__":
    test_crossing_box_is_not_a_duplicate()
    test_clipped_partial_view_is_still_a_duplicate()
    test_same_object_high_iou_is_a_duplicate()
    test_disjoint_box_survives()
    test_fused_predict_keeps_the_climber()
    test_humans_never_enter_tiled_or_fused_tracking()
    print("\nALL PASS")
