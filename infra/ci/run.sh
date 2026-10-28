#!/bin/bash
# Run the repository's complete test gate without GitHub Actions.
#
# This is intentionally a plain shell runner so it can run from the Mac mini
# over SSH or from launchd. It exits on the first failed phase and leaves a
# timestamped log under CI_LOG_DIR (or ~/logs/monkey-ci).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
UV="${UV_BIN:-$HOME/.local/bin/uv}"
CI_LOG_DIR="${CI_LOG_DIR:-$HOME/logs/monkey-ci}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$CI_LOG_DIR/ci-$STAMP.log"

mkdir -p "$CI_LOG_DIR"
export PATH="$HOME/.local/bin:$HOME/.local/node/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ ! -x "$UV" ]]; then
  echo "uv not found: $UV" >&2
  exit 2
fi
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  echo "node/npm not found; install Node 20 on the runner first" >&2
  exit 2
fi

run_checks() {
  set -e
  echo "== Monkey Detection local CI =="
  echo "started=$STAMP"
  echo "repo=$REPO_DIR"
  echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
  echo "node=$(node --version) npm=$(npm --version) uv=$("$UV" --version)"

  run_backend_test() {
  local name="$1"
  echo "::group::$name"
  "$UV" run --project "$REPO_DIR/backend" python "$REPO_DIR/tests/$name.py"
  echo "::endgroup::"
  }

  cd "$REPO_DIR"
  for test_name in \
  test_animal_groups \
  test_sdk_surface \
  test_white_balance \
  test_native_imgsz \
  test_detection_events \
  test_detection_lock \
  test_trail_video \
  test_second_stage \
  test_worker_resilience \
  test_idle_gate \
  test_rebuild_apply \
  test_rebuild_reference_media \
  test_backfill_consumer \
  test_reid_labels \
  test_platform_smoke; do
    run_backend_test "$test_name"
  done

  # Run fixture-based suites through pytest; script entry points do not collect them.
  echo "::group::pytest suites"
  "$UV" run --project "$REPO_DIR/backend" --with pytest python -m pytest -q \
    "$REPO_DIR/tests/test_stream_metrics.py" \
    "$REPO_DIR/tests/test_canary.py" \
    "$REPO_DIR/tests/test_single_node_watchdog.py" \
    "$REPO_DIR/tests/test_local_storage.py" \
    "$REPO_DIR/tests/test_snapshot_media.py" \
    "$REPO_DIR/tests/test_trail_video_fps.py" \
    "$REPO_DIR/tests/test_poc_batch.py" \
    "$REPO_DIR/tests/test_tiling_fusion.py" \
    "$REPO_DIR/tests/test_normalize_review_ids.py" \
    "$REPO_DIR/tests/test_plan_reference_apply.py" \
    "$REPO_DIR/tests/test_poc_camera_access.py" \
    "$REPO_DIR/tests/test_poc_sources.py" \
    "$REPO_DIR/tests/test_prepare_reference_rebuild.py" \
    "$REPO_DIR/tests/test_rebuild_clips.py" \
    "$REPO_DIR/tests/test_rebuild_reference_media.py" \
    "$REPO_DIR/tests/test_track_validity.py" \
    "$REPO_DIR/tests/test_validate_reference_rebuild.py"
  echo "::endgroup::"

  echo "::group::app imports and routes register"
  "$UV" run --project "$REPO_DIR/backend" python - <<'PY'
import sys
sys.path.insert(0, "backend")
import main
paths = {getattr(route, "path", "") for route in main.app.routes}
required = {
    "/api/health",
    "/api/detections",
    "/api/detections/events",
    "/api/admin/detections/visibility-bulk",
    "/api/admin/detections/lock-bulk",
    "/api/admin/detections/media-bulk",
    "/api/admin/detections/{detection_id}/lock",
    "/api/admin/reid/pairs",
    "/api/admin/reid/labels",
    "/api/admin/training/models",
}
missing = sorted(required - paths)
assert not missing, "routes missing: %s" % missing
print("OK — %d routes, all required present" % len(paths))
PY
  echo "::endgroup::"

  echo "::group::frontend install"
  cd "$REPO_DIR/frontend"
  npm ci
  echo "::endgroup::"
  echo "::group::frontend typecheck"
  npx tsc --noEmit -p tsconfig.json
  echo "::endgroup::"
  echo "::group::frontend lint"
  npm run lint
  echo "::endgroup::"
  echo "::group::frontend build"
  npm run build
  echo "::endgroup::"

  echo "CI PASS"
  echo "log=$LOG"
}

set +e
run_checks 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
