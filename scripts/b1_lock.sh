#!/usr/bin/env bash
# Correctness lock for speed work: the B1 tracks DBM of a target order must keep its recorded order-insensitive hash
# (scripts/b1_locks.json). The reference output differs between platforms with identical inputs and versions, so a
# lock holds only on the platform it was taken on: the locks are keyed by platform (macos-arm64 so far), and on a
# platform without one the script says so instead of failing. Cross-platform checks use counts and bands. Runs the harness's run_candidate.py on box B1 (as scripts/b1_selfcheck.sh does), hashes
# the DBM with the harness's dbm_hash.py, and fails on any change. Run it before any timing.
# usage: [TEASAR_PYTHON=python] scripts/b1_lock.sh OUT_DIR [cpu|metal] [best|kimimaro] [--record]
#   --record writes the hash as the order's new lock instead of checking it
# TEASAR_B1_STAGE=run stops after the candidate run (a Metal run's GPU job); TEASAR_B1_STAGE=judge only hashes
set -euo pipefail
out=$1 device=${2:-cpu} order=${3:-kimimaro} mode=${4:-check}
repo=$(cd "$(dirname "$0")/.." && pwd)
# the parity harness is not part of this repo, so there is no default
H=${TEASAR_HARNESS:?set TEASAR_HARNESS to the parity harness directory (run_candidate.py, compare.py; dbm_hash.py one level up)}
# the same interpreter b1_selfcheck.sh runs on: TEASAR_PYTHON when set, else the repo's .venv
py=${TEASAR_PYTHON:-$repo/.venv/bin/python}
stage=${TEASAR_B1_STAGE:-all}
if [ "$stage" != judge ]; then
  rm -rf "$out"
  echo "--- tree: $(git -C "$repo" rev-parse HEAD) $(git -C "$repo" status --porcelain | wc -l | tr -d ' ') changed paths ---"
  [ -n "${TEASAR_PYTHON:-}" ] && echo "--- candidate: $("$py" -I -c 'import teasar; print(teasar.__version__, teasar.__file__)') ---"
  TEASAR_B1_STAGE=run "$repo/scripts/b1_selfcheck.sh" "$out" "$device" "" "$order" > "$out.log" 2>&1
fi
[ "$stage" = run ] && exit 0
got=$("$py" "$H/../dbm_hash.py" "$out/tracks.dbm")
locks=$repo/scripts/b1_locks.json
platform=$(uname -s | sed 's/Darwin/macos/' | tr A-Z a-z)-$(uname -m)
if [ "$mode" = --record ]; then
  "$py" - "$locks" "$platform" "$order" "$got" <<'PY'
import json, sys
path, platform, order, h = sys.argv[1:]
d = json.load(open(path))
d.setdefault(platform, {})[order] = h
json.dump(d, open(path, "w"), indent=2, sort_keys=True)
open(path, "a").write("\n")
PY
  echo "b1 lock $order recorded for $platform: $got"
  exit 0
fi
want=$("$py" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], {}).get(sys.argv[3], ''))" \
  "$locks" "$platform" "$order")
if [ -z "$want" ]; then
  echo "b1 lock $order on $device: no $platform lock recorded (got $got); check counts and bands instead"
  exit 2
elif [ "$got" = "$want" ]; then
  echo "b1 lock $order on $device ($platform): PASS ($got)"
else
  echo "b1 lock $order on $device ($platform): FAIL, got $got, locked $want"
  exit 1
fi
