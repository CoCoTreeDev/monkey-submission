"""Regression tests for the resumable mini backfill queue."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from backfill_consumer import (
    QueueStore,
    build_manifest,
    init_queue,
    queue_counts,
    timestamp,
    utc_now,
)  # noqa: E402


def reference() -> dict:
    return {
        "reference_clips": 2,
        "canonical_clips": 2,
        "canonical": [
            {"canonical_id": "clip-a", "canonical_trail_path": "a_trail.mp4"},
            {"canonical_id": "clip-b", "canonical_trail_path": "b_trail.mp4"},
        ],
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference_path = root / "reference.json"
        queue_path = root / "queue.json"
        reference_path.write_text(json.dumps(reference()), encoding="utf-8")
        state = init_queue(reference_path, queue_path, "run-test")
        assert queue_counts(state) == {"queued": 2, "running": 0, "completed": 0, "failed": 0, "total": 2}

        store = QueueStore(queue_path)
        first = store.claim("worker-a", lease_seconds=60)
        assert first and first["job_id"] == "clip-a"
        assert store.claim("worker-b", lease_seconds=60)["job_id"] == "clip-b"
        assert store.claim("worker-c", lease_seconds=60) is None

        result = {"clip_id": "clip-a", "rows": [{"track_id": 1}]}
        store.save_result("clip-a", "worker-a", result)
        # A retry after an upload error keeps the expensive inference result.
        failed = store.fail("clip-a", "worker-a", "upload timeout", retry=True)
        assert failed["status"] == "queued"
        retry = store.claim("worker-a", lease_seconds=60)
        assert retry and retry["result"] == result
        store.complete("clip-a", "worker-a", result)

        store.fail("clip-b", "worker-b", "bad zip", retry=False)
        final = store.snapshot()
        assert queue_counts(final) == {"queued": 0, "running": 0, "completed": 1, "failed": 1, "total": 2}
        manifest = build_manifest(final, {"output_dir": str(root)})
        assert manifest["status"] == "review"
        assert manifest["clip_count"] == 1
        assert manifest["failed_count"] == 1
        assert manifest["apply_allowed"] is False

        # A killed worker is recoverable after its lease expires.
        second_queue = root / "expired-queue.json"
        init_queue(reference_path, second_queue, "run-expired")
        expired_store = QueueStore(second_queue)
        claimed = expired_store.claim("dead-worker", lease_seconds=60)
        assert claimed and claimed["job_id"] == "clip-a"
        expired_store.mutate(lambda value: value["jobs"][0].update({
            "lease_expires_at": timestamp(utc_now() - timedelta(seconds=1)),
        }))
        recovered = expired_store.claim("replacement-worker", lease_seconds=60)
        assert recovered and recovered["job_id"] == "clip-a"

    print("ok backfill_consumer: claim, retry checkpoint, and damaged-clip skip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
