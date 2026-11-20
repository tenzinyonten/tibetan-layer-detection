#!/usr/bin/env bash
# One-shot Miniconda + Python train_env setup for Vast.ai Linux instances.
# Runs each step sequentially and waits for it to finish before continuing.
#
#   bash quotation/src/setup_quotation_train_env.sh
#
# Env overrides:
#   CONDA_DIR=$HOME/miniconda
#   ENV_NAME=train_env
#   PYTHON_VERSION=3.12
#   FLASH_ATTN_VERSION=2.8.3
#   SKIP_FLASH_ATTN=1     # skip the build; training falls back to SDPA

set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda}"
ENV_NAME="${ENV_NAME:-train_env}"
# 3.12, not 3.14: numpy==2.2.6 (pinned in requirements.txt) has no cp314 wheel
# on PyPI yet (cp310-cp313 only; cp314 wheels only start at numpy 2.3.2), so
# 3.14 would force a from-source numpy build on a fresh instance. Verified
# against PyPI's simple index directly, not assumed. 3.12 has full wheel
# coverage for every pin in requirements.txt.
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"
INSTALLER="${INSTALLER:-$HOME/Miniconda3-latest-Linux-x86_64.sh}"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() {
  echo ""
  echo ">>> $*"
}

step "1/10 Download Miniconda installer"
if [[ -f "$INSTALLER" ]]; then
  echo "    Installer already exists at $INSTALLER (skipping download)"
else
  curl -fsSL -o "$INSTALLER" "$MINICONDA_URL"
fi

step "2/10 Install Miniconda to $CONDA_DIR"
if [[ -x "$CONDA_DIR/bin/conda" ]]; then
  echo "    Miniconda already installed (skipping)"
else
  bash "$INSTALLER" -b -p "$CONDA_DIR"
fi

step "3/10 Initialize conda for bash"
"$CONDA_DIR/bin/conda" init bash

step "4/10 Load conda into this shell"
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh"

step "5/10 Accept Anaconda Terms of Service (main channel)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

step "6/10 Accept Anaconda Terms of Service (r channel)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

step "7/10 Create conda environment '$ENV_NAME' (Python $PYTHON_VERSION)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "    Environment '$ENV_NAME' already exists (skipping create)"
else
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

step "8/10 Activate environment and verify Python version"
conda activate "$ENV_NAME"
python --version
echo "    python: $(which python)"

step "9/10 Install pinned requirements (torch lands here)"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"
python -c "import torch; print('    torch', torch.__version__, 'cuda', torch.cuda.is_available())"

step "10/10 Install Flash Attention 2 ($FLASH_ATTN_VERSION)"
# Must be a SEPARATE pip pass, after torch, with build isolation off: the
# flash-attn build imports torch, and an isolated build environment cannot see
# it. Measured ~2.1x faster train steps on a 3090 in earlier runs.
#
# This compiles a CUDA extension and can take 10-30 minutes on a fresh box.
# If it fails, training still works — train.py falls back to SDPA and says so.
if [[ "$SKIP_FLASH_ATTN" == "1" ]]; then
  echo "    SKIP_FLASH_ATTN=1 — skipping. Training will use SDPA (~2x slower)."
elif python -c "import flash_attn" 2>/dev/null; then
  echo "    flash-attn already installed (skipping build)"
else
  echo "    Building flash-attn==$FLASH_ATTN_VERSION — this can take 10-30 min."
  if python -m pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation; then
    python -c "import flash_attn; print('    flash_attn', flash_attn.__version__)"
  else
    echo "    !! flash-attn build FAILED."
    echo "    !! This is not fatal: train.py falls back to SDPA automatically."
    echo "    !! Expect roughly 2x slower steps. Re-run this script to retry."
  fi
fi

echo ""
echo "=== Setup complete ==="
echo "In a new shell, run:"
echo "  source \"$CONDA_DIR/etc/profile.d/conda.sh\""
echo "  conda activate $ENV_NAME"
echo "  python quotation/src/train_quotation_model.py --prepare-only"
