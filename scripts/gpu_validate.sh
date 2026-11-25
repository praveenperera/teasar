#!/usr/bin/env bash
# The GPU suite under Metal's API and shader validation (A5); fails on any validation message
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p _scratch
exec scripts/metal_validate.sh -o "_scratch/validate-$(date +%Y%m%d-%H%M%S).log" -- scripts/gpu_tests.sh "$@"
