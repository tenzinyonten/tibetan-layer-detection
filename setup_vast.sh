#!/usr/bin/env bash
# setup_vast.sh — run once on a fresh Vast.ai instance.
#   bash setup_vast.sh
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# /workspace is the persistent volume on most Vast templates. Anything outside
# it can vanish when the instance is stopped.
mkdir -p /workspace/runs /workspace/data
cd /workspace

echo "== python deps =="
pip install -q --upgrade pip
pip install -q \
    "transformers>=4.48" \
    "datasets>=2.20" \
    "accelerate>=0.34" \
    "torch" \
    "numpy>=2.0" \
    huggingface_hub \
    scikit-learn

# flash-attn is optional. It roughly halves memory at 8192 tokens, but it
# compiles slowly and breaks on mismatched CUDA/torch. Skip on failure —
# common/train_layer.py falls back to sdpa automatically.
echo "== flash-attn (optional, ~5-10 min; failure is non-fatal) =="
pip install -q flash-attn --no-build-isolation || \
    echo "  flash-attn unavailable; will use sdpa"

echo "== sanity =="
python - <<'PY'
import torch, transformers, datasets
print("torch       ", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets    ", datasets.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu         ", p.name, f"{p.total_memory/1e9:.0f} GB")
    print("bf16        ", torch.cuda.is_bf16_supported())
try:
    import flash_attn; print("flash-attn  ", flash_attn.__version__)
except ImportError:
    print("flash-attn   not installed (sdpa fallback)")
PY

echo
echo "Next:"
echo "  huggingface-cli login      # dataset is private"
echo "  python common/train_layer.py --smoke-test --output-dir /workspace/runs/smoke"
