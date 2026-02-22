---
language:
  - en
tags:
  - sdxl
  - lora
  - kaggle
  - workflow
library_name: diffusers
pipeline_tag: text-to-image
license: mit
---

# mihai-lora-v2 (workflow-only)

This repository contains the reproducible training workflow used to run a personal SDXL LoRA pipeline on free Kaggle GPU, including checkpoint continuation, checkpoint evaluation, and LinkedIn-style gallery generation.

No personal training images, captions, generated portraits, or model checkpoints are included.

## Included

- Kaggle training script (`07_kaggle/train_flux_lora.py`)
- Config/run automation scripts (`03_configs/*`)
- Evaluation script templates (`08_kaggle_eval/*`)
- Runbook and process notes

## Excluded

- Raw/curated personal photos
- Captions tied to personal data
- Checkpoints and model weights
- Generated output galleries
- API tokens and credentials

## Privacy note

If you publish similar work, keep biometric data and personal LoRA weights private unless you explicitly want public distribution.
