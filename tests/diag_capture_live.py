"""Live diagnosis of the production capture starvation, run ON the edge node.

Reads the running service's own /api/admin/inference-stats (the edge build
already reports capture_fps/reconnects locally even while the cloud schema
still filters them), counts RTSP sessions held by this machine, then restarts
the service and tracks the first three minutes: if capture starts at ~30/s and
decays to 3-4/s, the decay shape and timing point at the cause.
"""
import subprocess
import os
import sys
import time

import httpx

BASE = "http://localhost:8000"


def login() -> dict:
    tok = httpx.post(f"{BASE}/api/auth/login", json={"code": os.getenv("SEED_ADMIN_CODE", "demo-admin")},
                     timeout=10).json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def stats(label: str, headers: dict) -> None:
    d = httpx.get(f"{BASE}/api/admin/inference-stats", headers=headers, timeout=10).json()
    cells = []
    for c in d["cameras"]:
        idle = round((c.get("idle_ratio") or 0) * 100)
        cells.append(f"{c['camera_id']} 捕{c.get('capture_fps')} 消{c['sample_fps']}"
                     f" 空{idle}% 連{c.get('stream_reconnects')} 計{c.get('frames_processed')}")
        gaps = c.get("capture_gaps_ms") or {}
        if gaps:
            cells[-1] += f" 間隔p50={gaps.get('p50')}ms p90={gaps.get('p90')}ms max={gaps.get('max')}ms"
        st = c.get("stage_sec") or {}
        fp = c.get("frames_processed") or 0
        if st and fp:
            detail = " ".join(f"{k}={v / fp * 1000:.0f}ms" for k, v in
                              sorted(st.items(), key=lambda x: -x[1]))
            cells[-1] += f" [{detail}]"
    print(f"» {label:>12} | " + " | ".join(cells), flush=True)


def rtsp_sessions() -> str:
    out = subprocess.run(
        ["bash", "-c", "netstat -an | grep -E '\\.554\\s' | grep ESTABLISHED | wc -l"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


headers = login()
stats("運轉數小時", headers)
print(f"mini 持有的 ESTABLISHED RTSP(554) 連線數: {rtsp_sessions()}", flush=True)

print("=== 重啟 edge 服務，追蹤前三分鐘 ===", flush=True)
# kickstart -k silently failed to restart from the runner once (2026-07-30
# 07:48 deploy left the old build running) — bootout+bootstrap is the hard
# restart that has worked from this context every time.
subprocess.run(["bash", "-c",
                "launchctl bootout gui/$(id -u)/com.cocotree.monkey-edge 2>/dev/null; sleep 3; "
                "launchctl bootstrap gui/$(id -u) $HOME/Library/LaunchAgents/com.cocotree.monkey-edge.plist"])
time.sleep(25)

for _ in range(20):
    try:
        headers = login()
        break
    except Exception:
        time.sleep(3)
else:
    print("服務沒有回來 — 中止", flush=True)
    sys.exit(1)

t0 = time.time()
for _ in range(12):
    try:
        stats(f"t+{time.time() - t0:4.0f}s", headers)
    except Exception as exc:
        print(f"讀取失敗: {exc}", flush=True)
    time.sleep(15)
print(f"結束時 RTSP 連線數: {rtsp_sessions()}", flush=True)
