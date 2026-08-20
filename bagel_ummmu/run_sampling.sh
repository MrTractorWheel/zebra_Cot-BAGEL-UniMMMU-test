#!/usr/bin/env bash
# Launch Uni-MMMU sampling with Bagel-Zebra-CoT.
# Any extra args are forwarded to run_sampling.py (e.g. --task maze --limit 5).
#
# Typical molab session (12h limit):
#   bash run_sampling.sh --task all --time-budget-hours 11
# Next session (resumes automatically, finished cases are skipped):
#   bash run_sampling.sh --task all --time-budget-hours 11
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"

CKPT_DIR="${CKPT_DIR:-$WORK_DIR/ckpt/Bagel-Zebra-CoT}"
BAGEL_REPO="${BAGEL_REPO:-$WORK_DIR/Bagel-Zebra-CoT}"
UMMMU_ROOT="${UMMMU_ROOT:-$WORK_DIR/Uni-MMMU}"
MODEL_NAME="${MODEL_NAME:-bagel-zebra-cot}"

LOG="$WORK_DIR/sampling_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

python -u "$SCRIPT_DIR/run_sampling.py" \
  --checkpoint-dir "$CKPT_DIR" \
  --bagel-repo "$BAGEL_REPO" \
  --ummmu-root "$UMMMU_ROOT" \
  --model-name "$MODEL_NAME" \
  "$@" 2>&1 | tee "$LOG"
