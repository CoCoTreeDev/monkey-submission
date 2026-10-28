#!/bin/bash
# Sync model weights from the Azure Blob "models" container into ../models/.
# Large .pt files (e.g. the YOLO26x cascade verifier) exceed GitHub's 100 MB
# limit, so blob storage is the model registry; size-compare skips unchanged
# files so no-model deploys cost nothing.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MONKEY_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
    echo "uv not found; install it or set UV_BIN" >&2
    exit 2
fi
cd "$REPO/backend"
"$UV" run python - <<'PY'
import os
from pathlib import Path

from dotenv import load_dotenv

# Explicit path: find_dotenv() frame-walks and crashes when run via `python -`
load_dotenv(".env")
conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
if not conn:
    print("[models] no connection string — skipping sync")
    raise SystemExit(0)

from azure.storage.blob import ContainerClient

out = Path("../models")
out.mkdir(exist_ok=True)
cc = ContainerClient.from_connection_string(conn, "models")
try:
    blobs = list(cc.list_blobs())
except Exception as e:
    print(f"[models] container unavailable ({e!r}) — skipping sync")
    raise SystemExit(0)

for b in blobs:
    dst = out / b.name
    if dst.exists() and dst.stat().st_size == b.size:
        continue
    print(f"[models] downloading {b.name} ({b.size / 1048576:.0f} MB)")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        cc.download_blob(b.name).readinto(f)
    tmp.rename(dst)
print("[models] sync done")
PY
