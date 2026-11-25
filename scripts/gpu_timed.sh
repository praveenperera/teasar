#!/usr/bin/env bash
# Timed job step (medium priority). Part A: A7 unprofiled, a profiled pass, interim A8, validation negative control.
# A10, the fused extract benchmark the B9 projection uses, unprofiled then profiled, on the default host threads
# (the performance cores) and on 8 (TEASAR_CPU_THREADS, B9's gate row). Part B (B9): skeletonize on all 384 B1 slabs unprofiled three times, per-kernel and per-bucket profiles, a sweep of
# the host-piece threshold and batch size with peak memory
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/bench_prep.py gpu
TEASAR_PROFILE=1 python scripts/bench_prep.py gpu
/usr/bin/time -l python scripts/mem_prep.py 2>&1 | grep -E 'high water|peak memory footprint|maximum resident'
echo "--- validation negative control (expect FAIL) ---"
scripts/metal_validate.sh -o _scratch/oob-probe.log -- python -c "import teasar as tm; tm.Device()._oob_probe(); print('probe ran')"
echo "negative control exit $?"

echo "--- A10: extract, all 384 B1 slabs (the B9 projection) ---"
/usr/bin/time -l python scripts/bench_extract.py gpu 2>&1 | grep -vE "^ +[0-9]+  (involuntary|voluntary|signals|messages|block|page|swaps|instructions|cycles)"
TEASAR_PROFILE=1 python scripts/bench_extract.py gpu
echo "--- A10 on 8 host threads (Kimimaro's 8 workers) ---"
TEASAR_CPU_THREADS=8 python scripts/bench_extract.py gpu
TEASAR_CPU_THREADS=8 TEASAR_PROFILE=1 python scripts/bench_extract.py gpu 2>&1 | grep -E "host pieces|profiled pass|  pass"
echo "--- B9: skeletonize, all 384 B1 slabs ---"
for i in 1 2 3; do python scripts/prof_skel.py gpu; done
TEASAR_PROFILE=1 python scripts/prof_skel.py gpu
TEASAR_PROFILE=1 TEASAR_PROFILE_BUCKETS=1 python scripts/prof_skel.py gpu 2>&1 | grep -E "by piece size|slabs on"
for cp in 300 3000 10000; do echo "TEASAR_CPU_PIECE=$cp"; TEASAR_CPU_PIECE=$cp python scripts/prof_skel.py gpu; done
for bv in 8388608 33554432 134217728; do
  echo "TEASAR_SKEL_BATCH_VOXELS=$bv"
  TEASAR_SKEL_BATCH_VOXELS=$bv /usr/bin/time -l python scripts/prof_skel.py gpu 2>&1 | grep -E "slabs on|peak memory footprint"
done
exit 0
