#!/bin/bash
# Start the locally checked-out backend. Configuration comes from backend/.env.
set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MONKEY_REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
    echo "uv not found; install it or set UV_BIN" >&2
    exit 2
fi
cd "$REPO/backend"
"$UV" sync
exec "$UV" run uvicorn main:app --host 127.0.0.1 --port 8000
