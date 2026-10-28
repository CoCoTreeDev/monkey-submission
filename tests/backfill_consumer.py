"""Resumable single-consumer runner for a Reference rebuild.

The queue is a small, human-readable JSON state file.  A file lock makes a
second accidental consumer harmless; a lease makes a killed consumer's job
available again; and the result is checkpointed before upload so a transient
Blob failure does not run MPS inference twice.

The consumer only writes the isolated ``rebuild-review/`` prefix.  It never
changes detections and every generated manifest keeps ``apply_allowed`` false.

Typical mini usage::

    uv run --project backend python tests/backfill_consumer.py init \
      --reference ~/rawframes-backfill/reference-20260830/reference_manifest_v5.json \
      --queue ~/rawframes-backfill/reference-rebuild-20260831/queue.json

    API=... TOKEN=... uv run --project backend python tests/backfill_consumer.py run \
      --queue ~/rawframes-backfill/reference-rebuild-20260831/queue.json \
      --archive-root ~/rawframes-backfill/zips \
      --trail-root ~/rawframes-backfill \
      --model ~/Monkey-Detection/backend/best.pt \
      --tracker ~/Monkey-Detection/backend/trackers/bytetrack-monkey.yaml \
      --output ~/rawframes-backfill/reference-rebuild-20260831 \
      --upload --skip-failed
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import socket
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


SCHEMA_VERSION = 1
DEFAULT_LEASE_SECONDS = 2 * 60 * 60
DEFAULT_HEARTBEAT_SECONDS = 30


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class QueueStore:
    """Small locked queue store with atomic transitions."""

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_name(f"{path.name}.lock")

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(f"queue does not exist: {self.path}")
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported backfill queue schema")
        return state

    @contextmanager
    def _locked(self, *, write: bool = False) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = self._load()
            try:
                yield state
            finally:
                if write:
                    _atomic_write(self.path, state)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def snapshot(self) -> dict[str, Any]:
        with self._locked() as state:
            return copy.deepcopy(state)

    def mutate(self, callback: Callable[[dict[str, Any]], Any]) -> Any:
        with self._locked(write=True) as state:
            return callback(state)

    def claim(
        self, worker: str, *, lease_seconds: int = DEFAULT_LEASE_SECONDS
    ) -> dict[str, Any] | None:
        now = utc_now()

        def transition(state: dict[str, Any]) -> dict[str, Any] | None:
            for job in state["jobs"]:
                if job["status"] != "running":
                    continue
                expires = parse_timestamp(job.get("lease_expires_at"))
                if expires and expires <= now:
                    job.update({
                        "status": "queued",
                        "worker": None,
                        "lease_expires_at": None,
                        "last_error": "worker lease expired; requeued",
                    })

            candidates = [job for job in state["jobs"] if job["status"] == "queued"]
            if not candidates:
                return None
            job = min(candidates, key=lambda item: (item["ordinal"], item["job_id"]))
            expires = now + timedelta(seconds=lease_seconds)
            job.update({
                "status": "running",
                "worker": worker,
                "attempts": int(job.get("attempts", 0)) + 1,
                "claimed_at": timestamp(now),
                "heartbeat_at": timestamp(now),
                "lease_expires_at": timestamp(expires),
            })
            return copy.deepcopy(job)

        return self.mutate(transition)

    def heartbeat(self, job_id: str, worker: str, *, lease_seconds: int) -> bool:
        def transition(state: dict[str, Any]) -> bool:
            for job in state["jobs"]:
                if job["job_id"] != job_id:
                    continue
                if job["status"] != "running" or job.get("worker") != worker:
                    return False
                now = utc_now()
                job["heartbeat_at"] = timestamp(now)
                job["lease_expires_at"] = timestamp(
                    now + timedelta(seconds=lease_seconds)
                )
                return True
            return False

        return self.mutate(transition)

    def save_result(self, job_id: str, worker: str, result: dict[str, Any]) -> None:
        def transition(state: dict[str, Any]) -> None:
            job = _owned_running_job(state, job_id, worker)
            job["result"] = copy.deepcopy(result)
            job["result_saved_at"] = timestamp()

        self.mutate(transition)

    def complete(self, job_id: str, worker: str, result: dict[str, Any]) -> None:
        def transition(state: dict[str, Any]) -> None:
            job = _owned_running_job(state, job_id, worker)
            job.update({
                "status": "completed",
                "result": copy.deepcopy(result),
                "completed_at": timestamp(),
                "lease_expires_at": None,
                "heartbeat_at": timestamp(),
            })

        self.mutate(transition)

    def fail(
        self,
        job_id: str,
        worker: str,
        error: str,
        *,
        retry: bool,
    ) -> dict[str, Any]:
        def transition(state: dict[str, Any]) -> dict[str, Any]:
            job = _owned_running_job(state, job_id, worker)
            attempts = int(job.get("attempts", 0))
            can_retry = retry and attempts < int(state.get("max_attempts", 1))
            job.update({
                "status": "queued" if can_retry else "failed",
                "worker": None,
                "lease_expires_at": None,
                "last_error": error,
                "failed_at": None if can_retry else timestamp(),
            })
            return copy.deepcopy(job)

        return self.mutate(transition)

    def mark_manifest_published(self) -> None:
        self.mutate(lambda state: state.update({
            "manifest_published_at": timestamp(),
        }))


def _owned_running_job(
    state: dict[str, Any], job_id: str, worker: str
) -> dict[str, Any]:
    for job in state["jobs"]:
        if job["job_id"] == job_id:
            if job["status"] != "running" or job.get("worker") != worker:
                raise RuntimeError(f"job is not owned by worker: {job_id}")
            return job
    raise KeyError(f"unknown job: {job_id}")


def _reference_summary(reference: dict[str, Any]) -> dict[str, Any]:
    return {
        key: reference.get(key)
        for key in (
            "reference_clips", "canonical_clips", "missing_sources",
            "review_required", "time_overlap_warnings",
        )
        if key in reference
    }


def init_queue(reference_path: Path, queue_path: Path, run_id: str | None) -> dict[str, Any]:
    if queue_path.exists():
        raise FileExistsError(f"queue already exists: {queue_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    canonical = reference.get("canonical") or []
    if not canonical:
        raise ValueError("Reference contains no canonical clips")
    jobs = []
    seen: set[str] = set()
    for ordinal, clip in enumerate(canonical, 1):
        job_id = str(clip.get("canonical_id") or "")
        if not job_id or job_id in seen:
            raise ValueError(f"Reference contains duplicate/missing canonical_id: {job_id!r}")
        seen.add(job_id)
        jobs.append({
            "job_id": job_id,
            "ordinal": ordinal,
            "clip": copy.deepcopy(clip),
            "status": "queued",
            "attempts": 0,
            "worker": None,
            "result": None,
            "last_error": None,
        })
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or f"reference-rebuild-{utc_now().strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": timestamp(),
        "reference_path": str(reference_path),
        "reference": _reference_summary(reference),
        "max_attempts": 2,
        "manifest_published_at": None,
        "jobs": jobs,
    }
    _atomic_write(queue_path, state)
    return state


def queue_counts(state: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in ("queued", "running", "completed", "failed")}
    for job in state.get("jobs", []):
        counts[job.get("status", "failed")] = counts.get(job.get("status", "failed"), 0) + 1
    counts["total"] = len(state.get("jobs", []))
    return counts


def build_manifest(state: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    jobs = sorted(state.get("jobs", []), key=lambda item: item["ordinal"])
    clips = [job["result"] for job in jobs if job.get("status") == "completed" and job.get("result")]
    failures = [
        {
            "clip_id": job["job_id"],
            "source_trail_path": job["clip"].get("canonical_trail_path"),
            "status": "failed",
            "error": job.get("last_error") or "job failed",
        }
        for job in jobs if job.get("status") == "failed"
    ]
    counts = queue_counts(state)
    finished = counts["queued"] == 0 and counts["running"] == 0
    return {
        "run_id": state["run_id"],
        "created_at": state["created_at"],
        "status": "review" if finished else "running",
        "source": source,
        "clips": clips,
        "clip_count": len(clips),
        "row_count": sum(len(clip.get("rows", [])) for clip in clips),
        "failures": failures,
        "failed_count": len(failures),
        "reference": state.get("reference", {}),
        "rawframes_required": True,
        "apply_allowed": False,
        "consumer": {
            "queue": "file-lock-v1",
            "counts": counts,
            "manifest_published_at": state.get("manifest_published_at"),
        },
    }


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    destination = output_dir / "reference_rebuild_manifest.json"
    _atomic_write(destination, manifest)
    return destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basename(value: str) -> str:
    return Path(value).name


def _add_backend_to_path() -> None:
    import sys

    candidates = [
        Path(__file__).resolve().parent.parent / "backend",
        Path.cwd() / "backend",
        Path.home() / "Monkey-Detection" / "backend",
    ]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


def upload_result(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Upload one completed clip to the isolated review prefix idempotently."""
    _add_backend_to_path()
    from services.storage import storage

    def upload_asset(value: str) -> str:
        name = _basename(value)
        source = output_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"review asset missing: {source}")
        storage.upload(name, source.read_bytes())
        return name

    published = copy.deepcopy(result)
    for field in ("output_path", "rawframes_path", "poster_path"):
        value = published.get(field) or ""
        if value:
            published[field] = upload_asset(value)
    for row in published.get("rows", []):
        for field in ("image_path", "annotated_path", "crop_path"):
            value = row.get(field) or ""
            if value:
                row[field] = upload_asset(value)
    return published


def upload_manifest(manifest: dict[str, Any]) -> None:
    _add_backend_to_path()
    from services.storage import storage

    encoded = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    storage.upload(f"rebuild-review/{manifest['run_id']}.json", encoded)
    storage.upload("rebuild-review/latest.json", encoded)


class HeartbeatThread(threading.Thread):
    def __init__(self, store: QueueStore, job_id: str, worker: str, lease_seconds: int, interval: int):
        super().__init__(name=f"backfill-heartbeat-{job_id}", daemon=True)
        self.store = store
        self.job_id = job_id
        self.worker = worker
        self.lease_seconds = lease_seconds
        self.interval = interval
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                if not self.store.heartbeat(self.job_id, self.worker, lease_seconds=self.lease_seconds):
                    return
            except Exception as exc:  # the main loop owns the real failure
                print(f"[heartbeat warning] {self.job_id}: {exc}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()


def run_consumer(args: argparse.Namespace) -> int:
    store = QueueStore(args.queue)
    worker = args.worker or f"{socket.gethostname()}-{os.getpid()}"
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    source: dict[str, Any] = {
        "model": str(args.model),
        "model_sha256": file_sha256(args.model),
        "device": "mps",
        "imgsz": args.imgsz,
        "conf": args.conf,
        "tracker": str(args.tracker),
        "reference": store.snapshot().get("reference_path"),
        "output_dir": str(output_dir),
    }

    def checkpoint() -> dict[str, Any]:
        state = store.snapshot()
        manifest = build_manifest(state, source)
        write_manifest(output_dir, manifest)
        return manifest

    while True:
        job = store.claim(worker, lease_seconds=args.lease_seconds)
        if job is None:
            manifest = checkpoint()
            if args.upload and manifest["status"] == "review" and not manifest["consumer"]["manifest_published_at"]:
                upload_manifest(manifest)
                store.mark_manifest_published()
                checkpoint()
            counts = queue_counts(store.snapshot())
            print(json.dumps({"worker": worker, **counts}, ensure_ascii=False), flush=True)
            if args.once or (
                not args.watch and counts["queued"] == 0 and counts["running"] == 0
            ):
                return 0 if counts["failed"] == 0 or args.skip_failed else 2
            time.sleep(args.poll_seconds)
            continue

        job_id = job["job_id"]
        heartbeat = HeartbeatThread(
            store, job_id, worker, args.lease_seconds, args.heartbeat_seconds
        )
        if model is None:
            try:
                import numpy as np
                from ultralytics import YOLO
                model = YOLO(str(args.model))
                model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=args.imgsz, device="mps", verbose=False)
                backend = getattr(model.predictor, "model", None)
                if backend is not None and hasattr(backend, "model"):
                    backend.model.half()
                    backend.fp16 = True
            except Exception as exc:
                # A model/dependency/MPS failure is systemic, not a damaged
                # clip. Release the claim and stop instead of hiding the
                # problem behind --skip-failed.
                store.fail(job_id, worker, f"model initialization failed: {exc}", retry=True)
                checkpoint()
                raise
        heartbeat.start()
        try:
            if job.get("result"):
                result = copy.deepcopy(job["result"])
                print(f"[{job_id}] resuming from result checkpoint", flush=True)
            else:
                from rebuild_reference import rebuild_clip
                result = rebuild_clip(
                    model, job["clip"], args.archive_root, args.trail_root,
                    output_dir, args.imgsz, args.conf, args.tracker,
                    int(job["ordinal"]), store.snapshot()["run_id"],
                )
                store.save_result(job_id, worker, result)
                checkpoint()
            if args.upload:
                try:
                    result = upload_result(result, output_dir)
                except Exception as exc:
                    raise RuntimeError(f"review asset upload failed: {exc}") from exc
            store.complete(job_id, worker, result)
            manifest = checkpoint()
            print(f"[{job['ordinal']}] completed {job_id}", flush=True)
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            error = f"{type(exc).__name__}: {exc}"
            failed = store.fail(job_id, worker, error, retry=not args.skip_failed)
            checkpoint()
            print(f"[{job['ordinal']}] failed {job_id}: {error}", flush=True)
            if failed["status"] == "failed" and not args.skip_failed:
                return 2
        finally:
            heartbeat.stop()
            heartbeat.join(timeout=2)
        if args.once:
            return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a queue from a Reference manifest")
    init.add_argument("--reference", type=Path, required=True)
    init.add_argument("--queue", type=Path, required=True)
    init.add_argument("--run-id")

    status = subparsers.add_parser("status", help="show queue counts")
    status.add_argument("--queue", type=Path, required=True)

    run = subparsers.add_parser("run", help="consume queued clips")
    run.add_argument("--queue", type=Path, required=True)
    run.add_argument("--archive-root", type=Path, required=True)
    run.add_argument("--trail-root", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--tracker", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--imgsz", type=int, default=2048)
    run.add_argument("--conf", type=float, default=0.35)
    run.add_argument("--worker")
    run.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    run.add_argument("--heartbeat-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS)
    run.add_argument("--poll-seconds", type=int, default=15)
    run.add_argument("--once", action="store_true", help="process at most one queued clip")
    run.add_argument("--watch", action="store_true", help="stay alive and wait for new queued clips")
    run.add_argument("--upload", action="store_true", help="upload isolated review assets")
    run.add_argument("--skip-failed", action="store_true", help="mark bad clips failed and continue")

    args = parser.parse_args()
    if args.command == "init":
        state = init_queue(args.reference, args.queue, args.run_id)
        print(json.dumps({"queue": str(args.queue), "run_id": state["run_id"], **queue_counts(state)}, ensure_ascii=False))
        return 0
    if args.command == "status":
        state = QueueStore(args.queue).snapshot()
        print(json.dumps({"queue": str(args.queue), "run_id": state["run_id"], **queue_counts(state)}, ensure_ascii=False))
        return 0
    if not args.model.is_file():
        raise SystemExit(f"model missing: {args.model}")
    return run_consumer(args)


if __name__ == "__main__":
    raise SystemExit(main())
