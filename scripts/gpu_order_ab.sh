#!/usr/bin/env bash
# Timed job step (medium): A10 on all 384 B1 slabs, target_order 'best' against 'kimimaro' on Metal, both on 8 host
# threads (B9's gate row), as ABBA runs of bench_extract.py (warm-up plus the median of 5 passes each). Each run
# prints the host load before and after, and /usr/bin/time gives its process CPU time for the cores-busy figure
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "--- tree: $(git rev-parse HEAD) $(git status --porcelain | wc -l | tr -d ' ') changed paths ---"
export TEASAR_CPU_THREADS=8
for order in best kimimaro kimimaro best; do
  echo "--- A10 target_order=$order, TEASAR_CPU_THREADS=8 ---"
  /usr/bin/time -l python scripts/bench_extract.py gpu "$order" 2>&1 |
    grep -E "device|load|warm-up|A10 median|real|maximum resident|peak memory footprint"
done
exit 0
