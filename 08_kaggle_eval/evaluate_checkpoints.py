#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


OUT_DIR = Path("/kaggle/working/eval_outputs")

CHECKPOINTS = [
    "mihai_lora_v2_000001200.safetensors",
    "mihai_lora_v2_000001400.safetensors",
    "mihai_lora_v2_000001500.safetensors",
]

PROMPTS = [
    "professional LinkedIn headshot of mihai, navy blazer, clean gray studio background, photorealistic",
    "corporate profile photo of mihai, white shirt and dark jacket, soft office blur background, realistic lighting",
    "executive headshot of mihai, slight smile, 85mm portrait style, natural skin texture",
]

SEEDS = [11, 42]


def build_pipe() -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    return pipe


def resolve_checkpoint_dir() -> Path:
    base = Path("/kaggle/input")
    if not base.exists():
        raise SystemExit("/kaggle/input missing")

    for ds in sorted([p for p in base.iterdir() if p.is_dir()]):
        if list(ds.glob("*.safetensors")):
            return ds

    raise SystemExit(
        "No checkpoint dataset with .safetensors found under /kaggle/input"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = resolve_checkpoint_dir()
    print(f"checkpoint_dir={checkpoint_dir}")

    pipe = build_pipe()

    summary = []
    for ckpt in CHECKPOINTS:
        ckpt_path = checkpoint_dir / ckpt
        if not ckpt_path.exists():
            print(f"skip_missing_checkpoint={ckpt_path}")
            continue

        pipe.unload_lora_weights()
        pipe.load_lora_weights(str(checkpoint_dir), weight_name=ckpt)

        ckpt_dir = OUT_DIR / ckpt.replace(".safetensors", "")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for p_idx, prompt in enumerate(PROMPTS, start=1):
            for seed in SEEDS:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    negative_prompt="uncanny face, plastic skin, distorted teeth, extra fingers, watermark, text",
                    width=1024,
                    height=1024,
                    num_inference_steps=30,
                    guidance_scale=7.0,
                    generator=gen,
                ).images[0]

                out_name = f"p{p_idx}_seed{seed}.png"
                out_path = ckpt_dir / out_name
                image.save(out_path)
                summary.append(
                    {
                        "checkpoint": ckpt,
                        "prompt_index": p_idx,
                        "seed": seed,
                        "file": str(out_path),
                    }
                )
                print(f"saved={out_path}")

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"total_images={len(summary)}")


if __name__ == "__main__":
    main()
