#!/usr/bin/env bash
# GPU smoke (medium job step): the whole suite on the Metal device (B7 capped at 24 slabs, so the CPU side stays short
# inside the lease), B1's structural count check on Metal, then one profiled A10 pass. Runs every part and fails if
# any part failed
cd "$(dirname "$0")/.."
# the commit under test, so a tag can be checked against this log
echo "--- tree: $(git rev-parse HEAD) $(git status --porcelain | wc -l | tr -d ' ') changed paths ---"
rc=0
TEASAR_B7_SLABS="${TEASAR_B7_SLABS:-24}" scripts/gpu_tests.sh || rc=1
source .venv/bin/activate
echo "--- B1 count check (Metal) ---"
python scripts/b1_counts.py gpu _scratch/teasar-impl/b1-counts-gpu.json | tail -30 || rc=1
echo "--- A10 profiled pass ---"
TEASAR_PROFILE=1 python scripts/bench_extract.py gpu || rc=1
exit $rc
