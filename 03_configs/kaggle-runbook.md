# Kaggle Runbook (Free-tier)

## Cost

- Kaggle notebooks and GPU quota are free-tier based.
- You do not pay by default.
- Limits apply (GPU availability, weekly/session quotas).

## One-time setup

1. Install Kaggle credentials:
   - Download `kaggle.json` from your Kaggle account.
   - Place at `~/.kaggle/kaggle.json`.
   - `chmod 600 ~/.kaggle/kaggle.json`
2. Export username:
   - `export KAGGLE_USERNAME="your-kaggle-username"`

## Launch flow

Run:

```bash
/Users/mihai/mihai-lora-v2/03_configs/kaggle_cli_workflow.sh
```

This will:

- Prepare Kaggle dataset assets from `replicate_bundle_v2.zip`.
- Create or version dataset `KAGGLE_USERNAME/mihai-lora-v2-data`.
- Push kernel `KAGGLE_USERNAME/mihai-flux-lora-v2`.

## Monitor job status

```bash
/Users/mihai/mihai-lora-v2/.venv/bin/kaggle kernels status KAGGLE_USERNAME/mihai-flux-lora-v2
```

## Kernel source

- `07_kaggle/train_flux_lora.py`
- Uses chunked training and resume detection.
- Writes outputs to `/kaggle/working/output`.
