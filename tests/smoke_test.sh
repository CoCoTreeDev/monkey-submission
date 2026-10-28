#!/usr/bin/env bash
# Smoke test for the Monkey Detection Platform.
#
# Verifies the deployed backend is healthy, can store/serve snapshots,
# and returns detection records. Intended to run after CI/CD deploy.
#
# Usage:
#   bash tests/smoke_test.sh <API_URL>              # keep test data (manual)
#   bash tests/smoke_test.sh <API_URL> --cleanup     # delete test data after (CI)
#
# Environment:
#   ADMIN_PASSWORD  (default: demo-admin)
#
# Exit codes:
#   0 = all checks passed
#   1 = one or more checks failed

set -euo pipefail

API_URL="${1:?Usage: smoke_test.sh <API_URL> [--cleanup]}"
API_URL="${API_URL%/}"  # strip trailing slash
PASSWORD="${ADMIN_PASSWORD:-demo-admin}"
CLEANUP=false
if [ "${2:-}" = "--cleanup" ]; then
  CLEANUP=true
fi

PASSED=0
FAILED=0
CREATED_IDS=()

# ── Helpers ────────────────────────────────────────────────────────────────

pass() { PASSED=$((PASSED + 1)); echo "  ✓ $1"; }
fail() { FAILED=$((FAILED + 1)); echo "  ✗ $1"; }

check_status() {
  local label="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then
    pass "$label (HTTP $actual)"
  else
    fail "$label (expected $expected, got $actual)"
  fi
}

wait_for_health() {
  local max_attempts=12
  local delay=15
  local status="000"
  for ((i = 1; i <= max_attempts; i++)); do
    status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$API_URL/api/health" || echo "000")
    if [ "$status" = "200" ]; then
      pass "GET /api/health (HTTP 200, attempt $i/$max_attempts)"
      return 0
    fi
    if [ "$i" -lt "$max_attempts" ]; then
      echo "  … attempt $i/$max_attempts: HTTP $status, retrying in ${delay}s"
      sleep "$delay"
    fi
  done
  fail "GET /api/health (expected 200, got $status after $max_attempts attempts)"
  return 1
}

cleanup() {
  if [ ${#CREATED_IDS[@]} -gt 0 ]; then
    echo ""
    echo "── Cleanup ──"
    local ids_json
    ids_json=$(printf '%s\n' "${CREATED_IDS[@]}" | python3 -c "import sys,json; print(json.dumps([int(x) for x in sys.stdin.read().split()]))")
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "$API_URL/api/admin/delete-selected" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"ids\": $ids_json}")
    if [ "$status" = "200" ]; then
      echo "  Cleaned up ${#CREATED_IDS[@]} test detection(s)"
    else
      echo "  Warning: cleanup returned HTTP $status"
    fi
  fi
}

# ── Generate test images ──────────────────────────────────────────────────

generate_test_images() {
  python3 -c "
import struct, zlib, io, sys, os

def make_png(width, height, r, g, b, text=''):
    \"\"\"Generate a minimal PNG with a solid color and optional text label.\"\"\"
    # Build raw pixel rows (uncompressed)
    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter byte
        for x in range(width):
            raw += bytes([r, g, b])

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    buf = b'\x89PNG\r\n\x1a\n'
    buf += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    buf += chunk(b'IDAT', zlib.compress(raw))
    buf += chunk(b'IEND', b'')
    return buf

# Convert PNG to JPEG via Pillow if available, otherwise use PNG as-is
try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False

def save_jpg(png_bytes, path):
    if has_pil:
        img = Image.open(io.BytesIO(png_bytes))
        img.save(path, 'JPEG', quality=90)
    else:
        # Fallback: save as PNG with .jpg extension (still valid for testing)
        with open(path, 'wb') as f:
            f.write(png_bytes)

tmpdir = sys.argv[1]

# 1) Raw snapshot — green tinted (forest scene mock)
raw = make_png(320, 240, 34, 120, 50)
save_jpg(raw, os.path.join(tmpdir, 'raw.jpg'))

# 2) Annotated — same but with red tint (simulating bbox overlay)
annot = make_png(320, 240, 180, 60, 40)
save_jpg(annot, os.path.join(tmpdir, 'annot.jpg'))

# 3) Crop — orange (simulating zoomed monkey)
crop = make_png(80, 80, 200, 140, 50)
save_jpg(crop, os.path.join(tmpdir, 'crop.jpg'))

print('Generated 3 test images')
" "$1"
}

# ── Main ──────────────────────────────────────────────────────────────────

echo "Smoke test: $API_URL"
echo ""

# 1) Health check (retry — container may still be cold-starting after deploy)
echo "── Health ──"
wait_for_health

# 2) Auth
echo ""
echo "── Auth ──"
LOGIN_RESP=$(curl -s -X POST "$API_URL/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"code\":\"$PASSWORD\"}")
TOKEN=$(echo "$LOGIN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")
if [ -n "$TOKEN" ]; then
  pass "POST /api/auth/login"
else
  fail "POST /api/auth/login (no token)"
  echo ""
  echo "RESULT: $PASSED passed, $FAILED failed"
  exit 1
fi

# 3) Generate + upload test images
echo ""
echo "── Upload mock snapshots ──"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
generate_test_images "$TMPDIR"

TS=$(date -u +"%Y-%m-%dT%H:%M:%S")
UID_TAG=$(python3 -c "import uuid; print(uuid.uuid4().hex[:8])")
BASE="smoketest_t0_${UID_TAG}"

for suffix in raw annot crop; do
  if [ "$suffix" = "raw" ]; then
    FILENAME="${BASE}.jpg"
  else
    FILENAME="${BASE}_${suffix}.jpg"
  fi
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$API_URL/api/admin/upload-snapshot" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@${TMPDIR}/${suffix}.jpg;filename=${FILENAME}")
  check_status "Upload $FILENAME" "200" "$STATUS"
done

# 4) Verify snapshot serving
echo ""
echo "── Verify snapshot serving ──"
# Blob deployments answer 302 (redirect to a signed blob URL) so the bytes
# bypass the 1 vCPU container; -L follows to the actual image. Local
# deployments still stream 200 directly — both end at 200.
for suffix in "" "_annot" "_crop"; do
  FILENAME="${BASE}${suffix}.jpg"
  STATUS=$(curl -sL -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" "$API_URL/api/snapshots/${FILENAME}")
  check_status "GET /api/snapshots/$FILENAME (follow redirect)" "200" "$STATUS"
done

# 4b) Snapshot serving as a BROWSER experiences it
#
# Regression guard for the 2026-08-04 incident: the dashboard's 檢視 button
# probed the API with a cross-origin fetch(HEAD) and fell back to
# public/snapshots/ when that failed. Once the API started 302-ing to blob
# storage the probe failed CORS on the second hop, so every click landed on
# the static path — which holds almost nothing — and users got 404 pages while
# the API itself was perfectly healthy. Byte-level checks on the API alone
# cannot catch that class of bug; these can.
echo ""
echo "── Snapshot serving, browser-shaped ──"
ORIGIN="${FRONTEND_ORIGIN:-http://localhost:3000}"

# 4b-1 a REAL runtime snapshot (not the mock uploaded above) must be servable:
# the mock proves the endpoint works, a real detection proves production data is
# reachable — the incident only showed up on real files.
# Sample several recent detections and take the first whose file actually
# resolves: a single newest-record sample is racy — the daily orphan sweep and
# the user's own false-positive deletions can remove that file mid-test, which
# fails the check for a reason that has nothing to do with serving.
REAL=""
CANDIDATES=$(curl -s "$API_URL/api/detections?limit=10" -H "Authorization: Bearer $TOKEN" |
  python3 -c "
import sys, json
d = json.load(sys.stdin)
for i in d.get('items') or []:
    if i.get('image_path'):
        print(i['image_path'])")
while IFS= read -r CAND; do
  [ -z "$CAND" ] && continue
  CENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$CAND")
  CSTATUS=$(curl -sL -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer $TOKEN" "$API_URL/api/snapshots/${CENC}")
  if [ "$CSTATUS" = "200" ]; then REAL="$CAND"; break; fi
done <<< "$CANDIDATES"
if [ -z "$REAL" ] && [ -n "$CANDIDATES" ]; then
  fail "none of the 10 most recent detections has a servable snapshot"
fi
if [ -n "$REAL" ]; then
  ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$REAL")
  read -r STATUS CTYPE SIZE <<<"$(curl -sL -o /dev/null     -w "%{http_code} %{content_type} %{size_download}"     -H "Authorization: Bearer $TOKEN" "$API_URL/api/snapshots/${ENC}")"
  check_status "GET real snapshot (follow redirect)" "200" "$STATUS"
  case "$CTYPE" in
    image/*) pass "content-type is $CTYPE" ;;
    *) fail "real snapshot served as '$CTYPE', expected image/*" ;;
  esac
  if [ "${SIZE:-0}" -gt 10000 ]; then
    pass "payload ${SIZE} bytes"
  else
    fail "real snapshot only ${SIZE} bytes — redirect or blob likely broken"
  fi

  # 4b-2 CORS must survive the redirect, or any fetch()-based flow (probes,
  # preloads, canvas reads) silently fails and the UI falls back to nothing.
  ACAO=$(curl -s -D - -o /dev/null -H "Origin: $ORIGIN" \
    "$API_URL/api/snapshots/${ENC}" | tr -d '\r' |
    awk 'tolower($1) == "access-control-allow-origin:" {print $2}')
  if [ -n "$ACAO" ]; then
    pass "API redirect carries Access-Control-Allow-Origin: $ACAO"
  else
    fail "API snapshot response has no Access-Control-Allow-Origin — fetch-based UI paths will fail"
  fi
else
  echo "  (skipped: no detections in the DB to sample)"
fi

# 4b-3 source-level guard: no snapshot may be fetched through a probe that
# decides between the API and the static fallback. Checked in the SOURCE, not
# the built bundle — Next.js emits its own fetch(...,{method:"HEAD"}) for route
# prefetching, so grepping minified chunks flags the framework and tells us
# nothing. public/snapshots/ predates blob storage and holds almost nothing, so
# anything routed there 404s; that is exactly the 2026-08-04 incident.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/frontend/src"
if [ -d "$SRC" ]; then
  OFFENDERS=$(grep -rl 'method: *"HEAD"' "$SRC" 2>/dev/null || true)
  PROBES=""
  for F in $OFFENDERS; do
    # a HEAD request is only a problem when the same file also decides to fall
    # back to the static path — that pair is the broken pattern
    if grep -q 'snapshotFallbackUrl' "$F"; then PROBES="$PROBES $F"; fi
  done
  if [ -n "$PROBES" ]; then
    fail "snapshot HEAD-probe + static fallback pattern is back in:$PROBES"
  else
    pass "no snapshot HEAD-probe/static-fallback pattern in the frontend source"
  fi
else
  echo "  (skipped source guard: frontend/src not present)"
fi

# 5) Create a detection record
echo ""
echo "── Create test detection ──"
DET_RESP=$(curl -s -X POST "$API_URL/api/admin/create-detection" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"camera_id\": \"smoke-test\",
    \"timestamp\": \"$TS\",
    \"confidence\": 0.95,
    \"bbox_x\": 100, \"bbox_y\": 100, \"bbox_w\": 80, \"bbox_h\": 80,
    \"image_path\": \"${BASE}.jpg\",
    \"annotated_path\": \"${BASE}_annot.jpg\",
    \"crop_path\": \"${BASE}_crop.jpg\",
    \"track_id\": 0
  }")
DET_ID=$(echo "$DET_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")
if [ -n "$DET_ID" ] && [ "$DET_ID" != "None" ]; then
  pass "POST /api/admin/create-detection (id=$DET_ID)"
  CREATED_IDS+=("$DET_ID")
else
  fail "POST /api/admin/create-detection"
fi

# 6) Verify detection in list
echo ""
echo "── Verify detection list ──"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $TOKEN" "$API_URL/api/detections?camera_id=smoke-test&limit=1")
check_status "GET /api/detections?camera_id=smoke-test" "200" "$STATUS"

LIST_TOTAL=$(curl -s -H "Authorization: Bearer $TOKEN" "$API_URL/api/detections?camera_id=smoke-test&limit=1" | \
  python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo "0")
if [ "$LIST_TOTAL" -ge 1 ]; then
  pass "Detection appears in list (total=$LIST_TOTAL)"
else
  fail "Detection not in list (total=$LIST_TOTAL)"
fi

# 7) Storage stats
echo ""
echo "── Storage stats ──"
STATS=$(curl -s "$API_URL/api/admin/storage-stats" -H "Authorization: Bearer $TOKEN")
FILE_COUNT=$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_files',0))" 2>/dev/null || echo "0")
if [ "$FILE_COUNT" -ge 3 ]; then
  pass "Storage has $FILE_COUNT files (>= 3 test images)"
else
  fail "Storage has $FILE_COUNT files (expected >= 3)"
fi

# 8) Cleanup (only with --cleanup flag)
if [ "$CLEANUP" = true ]; then
  cleanup
else
  echo ""
  echo "── Skipping cleanup (pass --cleanup to remove test data) ──"
fi

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════"
echo " SMOKE TEST: $PASSED passed, $FAILED failed"
echo "════════════════════════════════════════"

if [ "$FAILED" -gt 0 ]; then
  exit 1
fi
