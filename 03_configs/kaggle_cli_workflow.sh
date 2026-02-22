#!/usr/bin/env bash
set -euo pipefail

VENV_KAGGLE="/Users/mihai/mihai-lora-v2/.venv/bin/kaggle"
ROOT="/Users/mihai/mihai-lora-v2"

if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
  echo "KAGGLE_USERNAME is missing"
  exit 1
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Missing auth. Set KAGGLE_API_TOKEN or install ~/.kaggle/kaggle.json"
  exit 1
fi

python3 "$ROOT/03_configs/prepare_kaggle_assets.py"

echo "Creating or updating Kaggle dataset..."
if "$VENV_KAGGLE" datasets status "${KAGGLE_USERNAME}/mihai-lora-v2-data" >/dev/null 2>&1; then
  "$VENV_KAGGLE" datasets version -p "$ROOT/07_kaggle/dataset" -m "Update LoRA v2 training zip"
else
  "$VENV_KAGGLE" datasets create -p "$ROOT/07_kaggle/dataset"
fi

echo "Pushing Kaggle kernel..."
"$VENV_KAGGLE" kernels push -p "$ROOT/07_kaggle"

echo "Kernel launched. Monitor with:"
echo "  $VENV_KAGGLE kernels status ${KAGGLE_USERNAME}/mihai-flux-lora-v2"
