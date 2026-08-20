#!/usr/bin/env bash
# Run the official Uni-MMMU evaluation (eval_ummmu.py) on the Bagel-Zebra-CoT outputs.
#
# IMPORTANT: the official judges are Qwen2.5-VL-72B-Instruct + Qwen3-32B in bf16,
# which need ~210 GB VRAM combined and do NOT fit on a single 96 GB GPU.
# On molab we substitute the official AWQ-quantized variants
# (~41 GB + ~19 GB, both fit together in 96 GB). Scores may differ slightly
# from a full-precision judge; note this when comparing against the paper.
#
# Run this in a SEPARATE session after sampling is finished:
#   bash run_eval.sh
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
UMMMU_ROOT="${UMMMU_ROOT:-$WORK_DIR/Uni-MMMU}"
MODEL_NAME="${MODEL_NAME:-bagel-zebra-cot}"

# Set USE_AWQ_JUDGES=0 on a machine with >=210GB VRAM (e.g. 3x A100/H100)
# to use the official full-precision judges.
USE_AWQ_JUDGES="${USE_AWQ_JUDGES:-1}"

cd "$UMMMU_ROOT"

# Evaluation dependencies (judges, metrics)
pip install qwen-vl-utils dreamsim cairosvg pandas xlsxwriter openpyxl autoawq

# eval_ummmu.py hardcodes base_path as a placeholder and the judge model names
# as dataclass defaults; patch a copy instead of editing the original.
cp eval_ummmu.py eval_ummmu_patched.py
sed -i "s#\"path-to-UMMMU-home\"#\"$UMMMU_ROOT\"#" eval_ummmu_patched.py
if [ "$USE_AWQ_JUDGES" = "1" ]; then
  sed -i 's#"Qwen/Qwen3-32B"#"Qwen/Qwen3-32B-AWQ"#' eval_ummmu_patched.py
  sed -i 's#"Qwen/Qwen2.5-VL-72B-Instruct"#"Qwen/Qwen2.5-VL-72B-Instruct-AWQ"#' eval_ummmu_patched.py
  echo "Patched eval_ummmu_patched.py to use AWQ judge models."
fi

LOG="$WORK_DIR/eval_$(date +%Y%m%d_%H%M%S).log"
echo "Evaluating outputs/$MODEL_NAME ; logging to $LOG"

python -u eval_ummmu_patched.py --model_name "$MODEL_NAME" 2>&1 | tee "$LOG"

echo ""
echo "Results: $UMMMU_ROOT/eval/$MODEL_NAME/all_tasks_summary_${MODEL_NAME}.xlsx"
