"""Safety checks for the low-priority, shape-preserving POC batch path."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

from services.detector import _FairLock, _POCBatchInference, _POCBatchRequest  # noqa: E402


def _request(shape: tuple[int, int, int]) -> _POCBatchRequest:
    return _POCBatchRequest(np.empty(shape, dtype=np.uint8), 1600, "mps", True, None)


def test_batcher_only_combines_identical_source_shapes() -> None:
    same = _request((720, 1280, 3))
    different = _request((240, 320, 3))
    assert _POCBatchInference._compatible(same, _request((720, 1280, 3)))
    assert not _POCBatchInference._compatible(same, different)


def test_one_camera_group_keeps_all_frames_shape_safe() -> None:
    frames = [np.empty((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
    grouped = _POCBatchRequest(
        frames[0], 1600, "mps", True, None, frames=frames,
    )
    assert len(grouped.frames) == 2
    assert _POCBatchInference._compatible(grouped, _request((720, 1280, 3)))
    assert not _POCBatchInference._compatible(grouped, _request((240, 320, 3)))


def test_core_waiter_is_served_before_poc_waiter() -> None:
    lock = _FairLock()
    lock.acquire()
    order: list[str] = []

    def wait(name: str, priority: int) -> None:
        lock.acquire(priority=priority)
        order.append(name)
        lock.release()

    poc = threading.Thread(target=wait, args=("poc", 1))
    poc.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with lock._mutex:
            if len(lock._waiters) == 2:
                break
        time.sleep(0.001)
    else:
        raise AssertionError("POC waiter did not reach the GPU queue")

    core = threading.Thread(target=wait, args=("core", 0))
    core.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with lock._mutex:
            if len(lock._waiters) == 3:
                break
        time.sleep(0.001)
    else:
        raise AssertionError("core waiter did not reach the GPU queue")

    lock.release()
    core.join(timeout=1)
    poc.join(timeout=1)
    assert order == ["core", "poc"]


def test_batch_tracker_receives_monkeys_only(monkeypatch) -> None:
    from types import SimpleNamespace
    import torch
    from ultralytics.engine.results import Results

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    seen = []

    class Model:
        def predict(self, frames, **kwargs):
            outputs = []
            for i, image in enumerate(frames):
                rows = torch.tensor([[1, 1, 20, 20, .9, 0], [25, 1, 50, 60, .99, 1]])[i:]
                if kwargs.get("classes") is not None:
                    rows = rows[torch.isin(rows[:, 5], torch.tensor(kwargs["classes"]))]
                outputs.append(Results(image, "", {0: "monkey", 1: "human"}, boxes=rows))
            return outputs

    tracker = SimpleNamespace(update=lambda result: seen.append(result.boxes.cls.tolist()))
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    batcher = _POCBatchInference("cpu", False)
    monkeypatch.setattr(batcher, "_model_for_request", lambda _: Model())
    request = _POCBatchRequest(frame, 640, "cpu", False, tracker, frames=[frame, frame])
    batcher._run_batch([request])
    assert seen == [[0.0], []], "mixed and human-only frames must never track humans"
    assert [len(result.boxes) for result in request.results] == [1, 0]
