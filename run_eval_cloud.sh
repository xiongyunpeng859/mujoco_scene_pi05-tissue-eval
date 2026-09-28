#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/cloud_env.sh"
out="${1:-$HOME/eval_ep01}"
mkdir -p "$out"
exec python -u pi05_scripts/test_jax_policy_rollout.py \
  --checkpoint "$CHECKPOINT" --known-round "$TEST_ROUND" \
  --output-dir "$out" --execute-chunk 50 --ticks 375
