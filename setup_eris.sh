#!/usr/bin/env bash
# Run once on the Eris A10G to set up the cpm (cell-phenotype MoA) environment.
# Usage:  bash setup_eris.sh
set -euo pipefail

echo "=== Eris A10G setup (cpm / cell-phenotype MoA) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

pip install -q --upgrade pip
# Core ML stack (CUDA 12 wheels)
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Vision backbones + classical ML for CV / calibration
pip install -q timm scikit-learn
# Image + data utils
pip install -q Pillow numpy pandas tqdm
# Only needed if the OpenPhenom (CA-MAE) backbone is used AND its license clears the rules:
#   pip install -q huggingface_hub transformers

echo "=== Package check ==="
python - <<'PY'
import torch, timm, sklearn, numpy, PIL
print(f"torch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
print(f"timm {timm.__version__}  sklearn {sklearn.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  VRAM: {p.total_memory/1e9:.0f}GB")
PY

echo "=== Extract data ==="
# Assumes public.zip is in the repo root (scp ~/Downloads/public.zip eris:/workspace/SobolevCellular/).
if [ -f "public.zip" ]; then
    python -m src.preprocess --zip public.zip --out data/
    echo "Data extracted + indexed under data/"
else
    echo "WARNING: public.zip not found in repo root — copy it here and re-run."
fi

echo "=== Setup complete. Typical run: ==="
echo "  python -m src.preprocess   --zip public.zip --out data/"
echo "  python -m src.train        --backbone imagenet_convnext_tiny --folds 5"
echo "  python -m src.calibrate    --oof oof_imagenet_convnext_tiny.npz"
echo "  python -m src.infer        --backbone imagenet_convnext_tiny --out submission.csv"
