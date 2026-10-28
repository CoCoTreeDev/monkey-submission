#!/bin/bash
# Export best_coreml_640.mlpackage from best.pt for ANE tiled inference.
# Runs in an ephemeral pinned env isolated from the project (--no-project):
# coremltools 8.2 needs numpy<2 and torch 2.4.x at EXPORT time only — the
# runtime backend keeps its own (newer) pins and just loads the mlpackage.
# Skips cheaply when the export already exists and is newer than best.pt.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MONKEY_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
    echo "uv not found; install it or set UV_BIN" >&2
    exit 2
fi
cd "$REPO/backend"
PT=best.pt
OUT=best_coreml_640.mlpackage
# The ANE paths are opt-in and currently disabled (they did not beat the
# full-frame baseline under cross-validation). Skip the
# expensive pinned-env export unless a camera is actually configured for them.
if ! grep -qE '^ANE_(TILE|FUSION)_CAMERAS=.+' .env 2>/dev/null; then
    echo "[coreml] no ANE cameras configured — skipping export"
    exit 0
fi
if [ -d "$OUT" ] && [ "$OUT" -nt "$PT" ]; then
    echo "[coreml] $OUT up to date"
    exit 0
fi
echo "[coreml] exporting $OUT from $PT"
rm -rf best.mlpackage  # drop stale leftovers from an interrupted run
if "$UV" run --no-project \
    --with "torch==2.4.1" --with "torchvision==0.19.1" --with "coremltools==8.2" \
    --with "numpy<2" --with "ultralytics==8.4.14" \
    python -c "from ultralytics import YOLO; YOLO('$PT').export(format='coreml', imgsz=640)" \
    && [ -d best.mlpackage ]; then
    rm -rf "$OUT"
    mv best.mlpackage "$OUT"
    echo "[coreml] wrote $OUT"
else
    echo "[coreml] export FAILED — keeping existing $OUT" >&2
    rm -rf best.mlpackage
    exit 1
fi
