#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is missing"
  exit 1
fi

KERNEL_REF="mihaichindris/mihai-flux-lora-v2"
OUT_DIR="/Users/mihai/mihai-lora-v2/07_kaggle/_kaggle_output_latest"
KAGGLE_BIN="/Users/mihai/mihai-lora-v2/.venv/bin/kaggle"

echo "Monitoring ${KERNEL_REF}..."
while true; do
  STATUS_LINE=$("$KAGGLE_BIN" kernels status "$KERNEL_REF")
  echo "$(date '+%Y-%m-%d %H:%M:%S')  $STATUS_LINE"

  if [[ "$STATUS_LINE" == *"RUNNING"* || "$STATUS_LINE" == *"QUEUED"* ]]; then
    sleep 45
    continue
  fi

  mkdir -p "$OUT_DIR"
  "$KAGGLE_BIN" kernels output "$KERNEL_REF" -p "$OUT_DIR" || true
  echo "Run finished with status: $STATUS_LINE"
  echo "Outputs (if any) downloaded to: $OUT_DIR"
  break
done
