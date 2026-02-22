# Colab Free Runbook (Quality-first, no paid plan)

This runbook assumes Colab paid plans are unavailable in your country.

## Decision

- Primary: Colab free tier with chunked resume-safe training.
- Fallback: Kaggle notebook continuation from checkpoint.
- Keep config stable for baseline quality comparison.

## Prereqs

- Prepared dataset: 36 image+caption pairs.
- Trigger token: `mihai`.
- Drive folder for persistence.

## Drive layout

- `MyDrive/mihai-lora-v2-colab/data`
- `MyDrive/mihai-lora-v2-colab/checkpoints`
- `MyDrive/mihai-lora-v2-colab/samples`
- `MyDrive/mihai-lora-v2-colab/logs`

## Baseline training settings

- Model: `black-forest-labs/FLUX.1-dev`
- Steps: `1600`
- Chunk size: `400`
- Learning rate: `0.00015`
- LoRA rank: `16`
- Resolution: `1024`
- Batch size: `1`
- Save every: `100`
- Validation sample every: `100`

## Chunk schedule

- Session A: `0 -> 400`
- Session B: `401 -> 800`
- Session C: `801 -> 1200`
- Session D: `1201 -> 1600`

Always resume from the latest checkpoint in Drive.

## Runtime rules

- Never store active checkpoints in `/content` only.
- After disconnect, reconnect runtime and resume.
- Do not change dataset/captions/hparams mid-baseline.

## Checkpoint selection

Evaluate checkpoints at `1000, 1200, 1400, 1600` using fixed prompts and seeds.
Pick best realism/likeness checkpoint, not necessarily the final step.

## If Colab GPU is unavailable

- Move to Kaggle notebook.
- Use the same dataset, prompts, seeds, and hyperparameters.
- Continue from last Drive checkpoint.
