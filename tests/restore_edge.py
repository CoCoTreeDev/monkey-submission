"""Restore the edge service after a failed bootstrap, with retries.

`launchctl bootstrap` intermittently fails with I/O error right after a
bootout (job still tearing down). Retry with backoff until the health
endpoint answers.
"""
import subprocess
import sys
import time

import httpx


def up() -> bool:
    try:
        return httpx.get("http://localhost:8000/api/health", timeout=5).status_code == 200
    except Exception:
        return False


if up():
    print("服務已在線上，無需處理", flush=True)
    sys.exit(0)

for attempt in range(8):
    subprocess.run(["bash", "-c",
                    "launchctl bootout gui/$(id -u)/com.cocotree.monkey-edge 2>/dev/null; sleep 4; "
                    "launchctl bootstrap gui/$(id -u) $HOME/Library/LaunchAgents/com.cocotree.monkey-edge.plist"],
                   capture_output=True, text=True)
    time.sleep(10)
    if up():
        print(f"服務已恢復（第 {attempt + 1} 次嘗試）", flush=True)
        sys.exit(0)
    print(f"第 {attempt + 1} 次嘗試失敗，重試 …", flush=True)
    time.sleep(5)

print("8 次嘗試後仍未恢復 — 需要人工介入", flush=True)
sys.exit(1)
