"""Does RTSP-over-TCP stop the lost-frame/reconnect storm?

Production streams (default UDP transport) reconnect every ~1-2 min per camera
and deliver 3-4 frames/s; fresh UDP sessions deliver 28-30/s but still show
h264 corruption warnings. UDP packet loss corrupting the h264 stream is the
prime suspect. This opens all cameras with rtsp_transport=tcp and measures
delivery in 30 s buckets plus reconnect count for 3 minutes. Green light for
production = steady ~30/s (~16-20 for 場域甲) and (almost) no reconnects.
"""
import logging
import os
import sys
import time

# Must be set before ffmpeg opens the stream — cv2 passes it per open().
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

sys.path.insert(0, os.path.expanduser("~/Monkey-Detection/backend"))
os.chdir(os.path.expanduser("~/Monkey-Detection/backend"))

import httpx
from dotenv import load_dotenv

load_dotenv(".env")
from services.stream import stream_manager

reconnects = {"n": 0}


class _CountReconnects(logging.Handler):
    def emit(self, record):
        if "reconnect" in record.getMessage():
            reconnects["n"] += 1


logging.getLogger("services.stream").addHandler(_CountReconnects())

api = os.getenv("CLOUD_API_URL")
tok = httpx.post(f"{api}/api/auth/login", json={"code": os.getenv("CLOUD_ACCESS_CODE")},
                 timeout=30).json()["token"]
cams = httpx.get(f"{api}/api/cameras", headers={"Authorization": f"Bearer {tok}"},
                 timeout=30).json()
ids = []
for c in cams:
    cid = c["camera_id"] + "-tcp"
    stream_manager.add(cid, c["rtsp_url"])
    ids.append(cid)
time.sleep(12)

connected = [cid for cid in ids if stream_manager.get(cid).connected]
print(f"TCP 連線成功 {len(connected)}/{len(ids)}", flush=True)
if not connected:
    print("cht.bovia 不接受 RTSP/TCP — 生產維持 UDP，需要別的解法", flush=True)
    sys.exit(1)

print(f"{'bucket':<8}" + "".join(f"{cid[:6]:>10}" for cid in ids), flush=True)
for b in range(6):
    start = {cid: stream_manager.get(cid)._seq for cid in ids}
    time.sleep(30)
    row = "".join(f"{(stream_manager.get(cid)._seq - start[cid]) / 30:>10.1f}" for cid in ids)
    print(f"{b * 30:>3}-{b * 30 + 30:<4}{row}", flush=True)

print(f"重連次數（3 分鐘, TCP）: {reconnects['n']}   ← 生產 UDP 約每 80 秒一次", flush=True)
stream_manager.stop_all()
