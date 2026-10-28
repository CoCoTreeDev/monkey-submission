"""Edge benchmark: where does the frame pipeline cap out, and does batching move it?

Production instrumentation (2026-07-30) shows workers finding no new frame on
93% of their GPU turns: cameras hand the detector 3-4 distinct frames/sec each
while a fresh capture probe sees 30/sec. Three stages isolate the cause:

  1. DELIVERY — fresh CameraStreams, delivered-FPS sampled in 30 s buckets for
     4 minutes. If fresh sessions start near 30/s and decay, the RTSP service
     throttles long-lived sessions; if they start low, the machine/process is
     the problem; if they stay high, production's stream objects are.
  2. PER-WORKER — the shipping loop shape (turn-first, fair lock, one predict
     per frame) on those same streams: distinct frames/sec.
  3. BATCHED — central gather-queue (the reverted bd20797 design): one thread
     runs predict() on same-shaped groups, workers block on their slot.
     Distinct frames/sec + realised batch size.

Both inference stages use predict() (no tracker) so the A/B compares GPU
dispatch strategies, not tracking overhead. Runs alongside production on
purpose — production load is part of the question.
"""
import os
import queue
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.expanduser("~/Monkey-Detection/backend"))
os.chdir(os.path.expanduser("~/Monkey-Detection/backend"))

import httpx
from dotenv import load_dotenv

load_dotenv(".env")

from config import INFERENCE_DEVICE, INFERENCE_IMGSZ, TRACKER_CONFIDENCE_THRESHOLD
import services.detector as det
from services.stream import stream_manager

api = os.getenv("CLOUD_API_URL")
tok = httpx.post(f"{api}/api/auth/login", json={"code": os.getenv("CLOUD_ACCESS_CODE")},
                 timeout=30).json()["token"]
cams = httpx.get(f"{api}/api/cameras", headers={"Authorization": f"Bearer {tok}"},
                 timeout=30).json()
ids = []
for c in cams:
    cid = c["camera_id"] + "-bench"
    stream_manager.add(cid, c["rtsp_url"])
    ids.append(cid)
time.sleep(12)

# ── stage 1: delivery decay, 8 buckets × 30 s ─────────────────────────────
print("== STAGE 1: fresh-session delivery (frames/sec, per 30 s bucket) ==", flush=True)
print(f"{'bucket':<8}" + "".join(f"{cid[:6]:>10}" for cid in ids), flush=True)
for b in range(8):
    start = {cid: stream_manager.get(cid)._seq for cid in ids}
    time.sleep(30)
    row = "".join(f"{(stream_manager.get(cid)._seq - start[cid]) / 30:>10.1f}" for cid in ids)
    print(f"{b * 30:>3}-{b * 30 + 30:<4}{row}", flush=True)

# ── shared harness for stages 2/3 ─────────────────────────────────────────
def run_workers(infer_one, label, dur=45):
    """4 threads consume via frame_after and call infer_one(cid, frame)."""
    stop = threading.Event()
    counts = {cid: 0 for cid in ids}
    lat = {cid: [] for cid in ids}

    def worker(cid):
        s = stream_manager.get(cid)
        seq = s._seq
        while not stop.is_set():
            frame, new_seq = s.frame_after(seq)
            if frame is None:
                time.sleep(0.002)
                continue
            seq = new_seq
            t0 = time.perf_counter()
            infer_one(cid, frame)
            lat[cid].append((time.perf_counter() - t0) * 1000)
            counts[cid] += 1

    threads = [threading.Thread(target=worker, args=(cid,), daemon=True) for cid in ids]
    [t.start() for t in threads]
    time.sleep(dur)
    stop.set()
    time.sleep(2)
    print(f"== {label} ({dur} s) ==", flush=True)
    agg = 0.0
    for cid in ids:
        fps = counts[cid] / dur
        agg += fps
        ms = statistics.median(lat[cid]) if lat[cid] else 0
        print(f"  {cid[:10]:<12} {fps:5.2f} 不同影格/s | {ms:5.1f} ms/呼叫", flush=True)
    print(f"  合計 {agg:.1f} 不同影格/s", flush=True)
    return agg


# ── stage 2: per-worker serialised predict (shipping shape) ───────────────
models = {cid: det._own_model() for cid in ids}

def per_worker(cid, frame):
    with det._gpu_lock:
        models[cid].predict(frame, imgsz=INFERENCE_IMGSZ, verbose=False,
                            conf=TRACKER_CONFIDENCE_THRESHOLD,
                            device=INFERENCE_DEVICE or None)

a2 = run_workers(per_worker, "STAGE 2: per-worker serialised")
del models

# ── stage 3: central batcher (reverted bd20797 design) ────────────────────
BATCH_MAX, GATHER_MS = 4, 12
shared = det._own_model()
reqs: "queue.Queue[tuple]" = queue.Queue()
running = True
batch_sizes = []

def batch_loop():
    while running:
        try:
            first = reqs.get(timeout=0.5)
        except queue.Empty:
            continue
        batch = [first]
        deadline = time.monotonic() + GATHER_MS / 1000
        while len(batch) < BATCH_MAX:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                nxt = reqs.get(timeout=left)
            except queue.Empty:
                break
            if nxt[1].shape == first[1].shape:
                batch.append(nxt)
            else:
                reqs.put(nxt)
                break
        batch_sizes.append(len(batch))
        shared.predict([r[1] for r in batch], imgsz=INFERENCE_IMGSZ, verbose=False,
                       conf=TRACKER_CONFIDENCE_THRESHOLD, device=INFERENCE_DEVICE or None)
        for r in batch:
            r[0].set()

bt = threading.Thread(target=batch_loop, daemon=True)
bt.start()

def batched(cid, frame):
    ev = threading.Event()
    reqs.put((ev, frame))
    ev.wait(timeout=30)

a3 = run_workers(batched, "STAGE 3: central batcher")
running = False
avg_b = statistics.mean(batch_sizes) if batch_sizes else 0
print(f"  實際平均批次大小 {avg_b:.2f}（上限 {BATCH_MAX}, 收集窗 {GATHER_MS} ms）", flush=True)
print(f"\n== VERDICT: per-worker {a2:.1f} vs batched {a3:.1f} 不同影格/s ==", flush=True)
stream_manager.stop_all()
