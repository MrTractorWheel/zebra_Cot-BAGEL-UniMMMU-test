#!/usr/bin/env bash
# One-time environment setup for running Bagel-Zebra-CoT on Uni-MMMU (molab, 1x RTX 6000 Pro Blackwell 96GB).
# Idempotent: safe to re-run after a session restart (skips finished steps).
#
# Usage:
#   export WORK_DIR=/path/to/persistent/storage   # optional, defaults to $HOME/bagel_ummmu
#   bash setup_molab.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/bagel_ummmu}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "== Work dir: $WORK_DIR =="
nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }

# ----------------------------------------------------------------------
# 1) Clone repos
# ----------------------------------------------------------------------
if [ ! -d Bagel-Zebra-CoT ]; then
  git clone https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT.git
fi
if [ ! -d Uni-MMMU ]; then
  git clone https://github.com/Vchitect/Uni-MMMU.git
fi

# ----------------------------------------------------------------------
# 2) Python dependencies
#    RTX 6000 Pro Blackwell is sm_120 -> needs torch built with CUDA 12.8+.
#    molab containers ship a Blackwell-ready torch already; only install torch
#    if it is missing (avoid downgrading/conflicting with the preinstalled one).
# ----------------------------------------------------------------------
python -c "import torch" 2>/dev/null || {
  echo "== torch not found; installing cu128 build =="
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
}
pip install -r "$SCRIPT_DIR/requirements_molab.txt"

# flash-attn: optional. BAGEL uses flash_attn_varlen_func; if the real package
# is unusable (no wheel for this torch/CUDA/Python combo, no nvcc to compile),
# the sampling code automatically falls back to a numerically-equivalent
# PyTorch SDPA shim (flash_attn_sdpa_shim.py), so this step never blocks setup.
if python -c "from flash_attn import flash_attn_varlen_func" 2>/dev/null; then
  echo "== flash-attn available =="
else
  echo "== flash-attn not usable; attempting install (non-fatal) =="
  MAX_JOBS="$(nproc)" TORCH_CUDA_ARCH_LIST="12.0+PTX" pip install flash-attn --no-build-isolation \
    || echo "== flash-attn install failed -> sampling will use the SDPA shim (fine) =="
fi

# ----------------------------------------------------------------------
# 3) Download the Bagel-Zebra-CoT checkpoint (~30 GB)
# ----------------------------------------------------------------------
CKPT_DIR="$WORK_DIR/ckpt/Bagel-Zebra-CoT"
if [ ! -f "$CKPT_DIR/ae.safetensors" ]; then
  echo "== Downloading Bagel-Zebra-CoT checkpoint to $CKPT_DIR =="
  HF_HUB_ENABLE_HF_TRANSFER=1 python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="multimodal-reasoning-lab/Bagel-Zebra-CoT",
    local_dir="$CKPT_DIR",
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
EOF
fi

# ----------------------------------------------------------------------
# 4) Download the Uni-MMMU-Eval dataset and extract into the Uni-MMMU repo
# ----------------------------------------------------------------------
if [ ! -d "$WORK_DIR/Uni-MMMU/data" ]; then
  echo "== Downloading Uni-MMMU-Eval dataset =="
  HF_HUB_ENABLE_HF_TRANSFER=1 python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Vchitect/Uni-MMMU-Eval",
    repo_type="dataset",
    local_dir="$WORK_DIR/Uni-MMMU-Eval",
    resume_download=True,
)
EOF
  tar -xf "$WORK_DIR/Uni-MMMU-Eval/data.tar" -C "$WORK_DIR/Uni-MMMU"
fi

# ----------------------------------------------------------------------
# 5) Sanity check: model imports work on this GPU
# ----------------------------------------------------------------------
python - <<EOF
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("compute capability:", cap)
try:
    from flash_attn import flash_attn_varlen_func
    print("attention backend: real flash-attn")
except Exception:
    print("attention backend: PyTorch SDPA shim (flash-attn not usable)")
x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).shape)
EOF

echo ""
echo "== Setup complete =="
echo "Checkpoint : $CKPT_DIR"
echo "Bagel repo : $WORK_DIR/Bagel-Zebra-CoT"
echo "Uni-MMMU   : $WORK_DIR/Uni-MMMU"
echo ""
echo "Smoke test (2 cases of the science task):"
echo "  bash $SCRIPT_DIR/run_sampling.sh --task science --limit 2"
