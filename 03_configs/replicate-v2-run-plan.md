# Replicate v2 run plan (FLUX.1-dev LoRA)

This run plan is designed for `ostris/flux-dev-lora-trainer` and keeps your old model untouched.

## Sources used

- Replicate trainer README: recommends 1000-3000 steps and high-res images (~1024).
- Replicate fast trainer README: supports subject/style mode, auto-captioning, and optional per-image `.txt` captions.
- Hugging Face FLUX QLoRA post (consumer fine-tuning focus).
- Recent community ComfyUI/ai-toolkit practice for identity LoRAs.

## Dataset targets

- 24-36 curated images.
- Keep visual variety: lighting, angles, outfits, backgrounds.
- Avoid low-quality, filtered, or heavily compressed images.
- Caption style: include trigger token in every caption.
- Current bundle: 36 image+caption pairs in `replicate_bundle_v2.zip`.

## Trigger token

- Primary token: `mihai`
- Keep exact token stable across all runs.

## Baseline run

- steps: 1600
- learning_rate: 0.00015
- rank: 16
- resolution: 1024
- batch_size: 1

Training type: subject

Rationale: lower LR than old v1 (`0.0004`) to reduce overfitting/plastic artifacts.

## Sweep matrix

Run A (identity-stable)
- steps: 1400
- learning_rate: 0.00012
- rank: 16

Run B (baseline)
- steps: 1600
- learning_rate: 0.00015
- rank: 16

Run C (capacity test)
- steps: 1800
- learning_rate: 0.00012
- rank: 32

Optional Run D (faster convergence check)
- trainer: replicate/fast-flux-trainer
- steps: 1400
- type: subject

## Selection criteria

- Face likeness at 100% zoom.
- Natural skin texture (no wax/plastic look).
- Eyes/teeth/ears symmetry and realism.
- Consistency across business prompts.

## Output naming

- model: `mihai-chindris/image-generator-v2`
- checkpoints: `v2-runA`, `v2-runB`, `v2-runC`
