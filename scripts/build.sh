#!/usr/bin/env bash
# Builds the Metal + testing wheel into the repo venv; a CPU task, never a GPU job step
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export CARGO_BUILD_JOBS=4
maturin develop --uv --release -F python,metal,testing
python -c "import teasar._core as c; assert c.TESTING"
