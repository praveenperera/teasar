#!/usr/bin/env bash
# GPU correctness run for a queue job step: a fast compile smoke, then the suite on the Metal device only
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TEASAR_GPU=1 TEASAR_DEVICES=gpu
python -c "
import numpy as np, teasar as tm
d = tm.Device()
print(d.name, d.prepare_labels(np.ones((8, 8, 8), np.uint8), 4, 6, 10000, settings='villa_default').tolist(), flush=True)
"
python -m pytest tests -q -rs "$@"
