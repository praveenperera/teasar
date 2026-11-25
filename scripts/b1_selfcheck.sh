#!/usr/bin/env bash
# B1 self-check: runs the harness's run_candidate.py on box B1 with a device, over spec part A §2.5's baseline
# partition, then judges it as spec part B 4.1 (rev 3.1) does: compare.py --band gates the exact checks
# (empty_mismatch, grid_violations, finite metrics) and reports the reference divergence against the noise band, and
# skel_counts.py gates the structural count check against the count band. Inputs come from the fixture root
# $TEASAR_FIXTURES (data/, ref/, villa/), and every step runs on the repo's own .venv. CPU runs are CPU tasks; Metal
# runs go through the queue
# usage: [TEASAR_PYTHON=python] scripts/b1_selfcheck.sh OUT_DIR [cpu|metal] [TEASAR_CHOICES] [best|kimimaro]
# TEASAR_B1_STAGE=run stops after the candidate run and judge only judges an earlier run, so a Metal run's GPU job
# leaves the judging to a CPU task
set -euo pipefail
out=$1 device=${2:-cpu}
[ $# -ge 3 ] && [ -n "$3" ] && export TEASAR_CHOICES=$3
# the target order of every extract call; one run never mixes the two
order=${4:-best}
# the fixture root and the parity harness directory are not part of this repo, so neither has a default
root=${TEASAR_FIXTURES:?set TEASAR_FIXTURES to the fixture root (the directory holding data/, ref/ and villa/)}
H=${TEASAR_HARNESS:?set TEASAR_HARNESS to the parity harness directory (run_candidate.py, compare.py, skel_counts.py)}
repo=$(cd "$(dirname "$0")/.." && pwd)
# the harness imports the candidate package by this name
export TEASAR_MODULE=${TEASAR_MODULE:-teasar}
# TEASAR_PYTHON runs every step on another interpreter, such as a clean venv with a release wheel installed; the
# repo's python/ then stays off the path, so the installed package is the one the harness imports
if [ -n "${TEASAR_PYTHON:-}" ]; then
  py=$TEASAR_PYTHON pypath=
else
  py=$repo/.venv/bin/python pypath=$repo/python
fi
mkdir -p "$out"
stage=${TEASAR_B1_STAGE:-all}
[ "$stage" != judge ] && env ${pypath:+PYTHONPATH=$pypath} "$py" "$H/run_candidate.py" --villa "$root/villa" \
  --pred "$root/data/b1box.zarr/0" --out "$out/tracks.dbm" --z-min 8068 --z-max 9600 \
  --yx-bounds 0 0 1535 1535 --device "$device" --target-order "$order" --counts "$out/counts.json" | tee "$out/bench.log"
[ "$stage" = run ] && exit 0
rc=0
"$py" "$H/compare.py" "$root/ref/mem2_exp2_w8.dbm" "$out/tracks.dbm" --band "$root/ref/band_frozen.json" \
  --json "$out/compare.json" > "$out/divergence.txt" || rc=1
echo "exactness gate: $([ $rc = 0 ] && echo PASS || echo FAIL) (divergence report in $out/divergence.txt)"
"$py" "$H/skel_counts.py" compare "$root/ref/counts_ref.json" "$out/counts.json" --band "$root/ref/count_band.json" \
  | tee "$out/counts.txt" || rc=1
exit $rc
