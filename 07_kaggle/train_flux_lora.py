#!/usr/bin/env python3
"""Kaggle training entrypoint for chunked FLUX LoRA runs.

Run this script inside a Kaggle Code notebook/job.
It expects:
- Training zip in /kaggle/input/<dataset>/replicate_bundle_v2.zip
- Optional previous checkpoints dataset mounted under /kaggle/input/<checkpoint-dataset>/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


def sh(cmd: str) -> None:
    print(f"[cmd] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def resolve_hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.getenv(key)
        if val:
            return val

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        client = UserSecretsClient()
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                val = client.get_secret(key)
                if val:
                    return val
            except Exception:
                pass
    except Exception:
        pass

    return None


def latest_ckpt(root: Path) -> tuple[str | None, int]:
    if not root.exists():
        return None, 0
    cands: list[tuple[int, Path]] = []
    for p in root.glob("**/*"):
        if p.is_file() and p.suffix in {".safetensors", ".pt", ".bin"}:
            m = re.search(r"(\d+)(?!.*\d)", p.stem)
            step = int(m.group(1)) if m else -1
            cands.append((step, p))
    if not cands:
        return None, 0
    step, path = sorted(cands, key=lambda x: x[0])[-1]
    return str(path), max(step, 0)


def find_training_zip(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    matches = glob.glob("/kaggle/input/*/replicate_bundle_v2.zip")
    if not matches:
        raise FileNotFoundError(
            "Could not find replicate_bundle_v2.zip in /kaggle/input"
        )
    return Path(matches[0])


def find_dataset_folder_with_pairs(root: str = "/kaggle/input") -> Path | None:
    base = Path(root)
    if not base.exists():
        return None

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    for ds in sorted([p for p in base.iterdir() if p.is_dir()]):
        images = [
            p for p in ds.iterdir() if p.is_file() and p.suffix.lower() in image_exts
        ]
        if not images:
            continue

        pairs = 0
        for img in images:
            if (ds / f"{img.stem}.txt").exists():
                pairs += 1

        if pairs >= 10:
            return ds

    return None


def write_config(
    out_path: Path,
    data_dir: Path,
    run_root: Path,
    trigger: str,
    rank: int,
    lr: float,
    end_steps: int,
) -> None:
    text = f"""
job: extension
config:
  name: mihai_lora_v2
  process:
    - type: sd_trainer
      training_folder: "{run_root}"
      device: cuda:0
      network:
        type: lora
        linear: {rank}
        linear_alpha: {rank}
      save:
        dtype: float16
        save_every: 100
        max_step_saves_to_keep: 30
      datasets:
        - folder_path: "{data_dir}"
          caption_ext: "txt"
          default_caption: "photo of {trigger}"
          resolution: [768, 896, 1024]
      train:
        batch_size: 1
        steps: {end_steps}
        lr: {lr}
        gradient_accumulation_steps: 4
        train_unet: true
        train_text_encoder: false
        noise_scheduler: ddim
        optimizer: adamw8bit
        dtype: fp16
      model:
        name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
        is_xl: true
        low_vram: true
"""
    out_path.write_text(text.strip() + "\n", encoding="utf-8")


def find_checkpoint_by_step(root: Path, step: int) -> Path | None:
    pattern = f"*{step:07d}.safetensors"
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def hydrate_checkpoints_from_resume(resume_root: Path, ckpt_dir: Path) -> int:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in resume_root.glob("**/*.safetensors"):
        dst = ckpt_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
            copied += 1
    return copied


def run_checkpoint_eval(ckpt_dir: Path, trigger: str) -> None:
    candidates: list[Path] = []
    for step in (1200, 1400, 1500, 1600):
        ck = find_checkpoint_by_step(ckpt_dir, step)
        if ck is not None:
            candidates.append(ck)

    final_ck = ckpt_dir / "mihai_lora_v2.safetensors"
    if final_ck.exists():
        candidates.append(final_ck)

    dedup: list[Path] = []
    seen = set()
    for c in candidates:
        if c.name not in seen:
            dedup.append(c)
            seen.add(c.name)
    candidates = dedup

    if not candidates:
        print("No eval checkpoints found; skipping eval.")
        return

    out_dir = ckpt_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    prompts = [
        f"professional LinkedIn headshot of {trigger}, navy blazer, clean gray studio background, photorealistic",
        f"corporate profile photo of {trigger}, white shirt and dark jacket, soft office blur background, realistic lighting",
        f"executive headshot of {trigger}, slight smile, 85mm portrait style, natural skin texture",
    ]

    summary = []
    for ckpt in candidates:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        pipe.load_lora_weights(str(ckpt_dir), weight_name=ckpt.name)
        ck_name = ckpt.stem
        ck_out = out_dir / ck_name
        ck_out.mkdir(parents=True, exist_ok=True)

        for idx, prompt in enumerate(prompts, start=1):
            seed = 42 + idx
            gen = torch.Generator(device="cpu").manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt="uncanny face, plastic skin, distorted teeth, asymmetrical eyes, watermark, text",
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=gen,
            ).images[0]
            out_path = ck_out / f"p{idx}_seed{seed}.png"
            image.save(out_path)
            summary.append(
                {
                    "checkpoint": ckpt.name,
                    "prompt": idx,
                    "seed": seed,
                    "file": str(out_path),
                }
            )
            print(f"eval_saved={out_path}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"eval_total_images={len(summary)}")


def run_linkedin_pack(ckpt_dir: Path, trigger: str) -> None:
    preferred_steps = (1400, 1500, 1200)
    selected: Path | None = None
    for step in preferred_steps:
        selected = find_checkpoint_by_step(ckpt_dir, step)
        if selected is not None:
            break

    if selected is None:
        final_ck = ckpt_dir / "mihai_lora_v2.safetensors"
        if final_ck.exists():
            selected = final_ck

    if selected is None:
        print("No checkpoint available for LinkedIn pack generation.")
        return

    out_dir = ckpt_dir / "linkedin_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        f"professional LinkedIn headshot of {trigger}, navy blazer, clean gray studio background, photorealistic",
        f"corporate profile portrait of {trigger}, white shirt and charcoal blazer, realistic office bokeh background",
        f"executive headshot of {trigger}, subtle confident smile, 85mm portrait style, natural skin texture",
        f"business profile image of {trigger}, modern office setting, polished attire, realistic studio lighting",
        f"LinkedIn profile portrait of {trigger}, direct eye contact, minimal background, crisp professional look",
        f"professional headshot of {trigger}, dark blazer, soft key light, true-to-life facial details",
        f"corporate portrait of {trigger}, balanced lighting, neutral backdrop, authentic skin tones",
        f"executive business headshot of {trigger}, approachable expression, clean composition, photorealistic",
        f"high-end LinkedIn portrait of {trigger}, medium close-up, realistic color grading, professional style",
        f"professional profile photo of {trigger}, office interior blur, natural expression, realistic details",
    ]
    seeds = (101, 202, 303)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(str(ckpt_dir), weight_name=selected.name)

    manifest: dict[str, object] = {"selected_checkpoint": selected.name, "images": []}
    images = []
    for p_idx, prompt in enumerate(prompts, start=1):
        for seed in seeds:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt="uncanny face, plastic skin, asymmetrical eyes, distorted teeth, watermark, text, cartoon",
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=gen,
            ).images[0]
            filename = f"p{p_idx:02d}_s{seed}.png"
            out_path = out_dir / filename
            image.save(out_path)
            entry = {"file": str(out_path), "prompt_index": p_idx, "seed": seed}
            images.append(entry)
            print(f"pack_saved={out_path}")

    manifest["images"] = images
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"linkedin_pack_total={len(images)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--total-steps", type=int, default=1500)
    parser.add_argument("--trigger", default="mihai")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--resume-root", default="/kaggle/input")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    eval_after_train = True

    print("Listing /kaggle/input:")
    sh("ls -la /kaggle/input || true")

    dataset_folder = find_dataset_folder_with_pairs("/kaggle/input")
    training_zip: Path | None = None
    if dataset_folder is None:
        training_zip = find_training_zip(args.input_zip)

    gpu_probe = subprocess.run(
        "nvidia-smi -L", shell=True, capture_output=True, text=True
    )
    if gpu_probe.returncode != 0:
        raise RuntimeError(
            "No GPU runtime detected. Enable GPU accelerator in Kaggle and complete account verification requirements."
        )
    print(gpu_probe.stdout.strip())

    data_dir = Path("/kaggle/working/data")
    run_root = Path("/kaggle/working/output")
    cfg_path = Path("/kaggle/working/train_chunk.yaml")
    toolkit_dir = Path("/tmp/ai-toolkit")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    if dataset_folder is not None:
        print(f"Using mounted dataset folder directly: {dataset_folder}")
        for item in dataset_folder.iterdir():
            if item.is_file():
                shutil.copy2(item, data_dir / item.name)
    else:
        assert training_zip is not None
        sh(f'python -m zipfile -e "{training_zip}" "{data_dir}"')

    if toolkit_dir.exists():
        shutil.rmtree(toolkit_dir)
    sh("git clone --depth 1 https://github.com/ostris/ai-toolkit /tmp/ai-toolkit")
    sh("python -m pip install -q -r /tmp/ai-toolkit/requirements.txt")
    sh(
        "python -m pip install -q accelerate bitsandbytes transformers diffusers safetensors"
    )

    hf_token = resolve_hf_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("HF token loaded from env or Kaggle secrets.")
    else:
        print(
            "HF token not found. If FLUX repo is gated, add HF_TOKEN in Kaggle Secrets."
        )

    resume_path, discovered_steps = latest_ckpt(Path(args.resume_root))
    start = max(args.start_step, discovered_steps)
    end = min(start + args.chunk_size, args.total_steps)

    ckpt_output_dir = run_root / "mihai_lora_v2"
    hydrated = hydrate_checkpoints_from_resume(Path(args.resume_root), ckpt_output_dir)
    print(f"hydrated_checkpoints={hydrated}")

    if start >= args.total_steps:
        print("All requested steps already completed.")
        if args.eval_only or eval_after_train:
            run_checkpoint_eval(ckpt_output_dir, args.trigger)
            run_linkedin_pack(ckpt_output_dir, args.trigger)
        return

    write_config(cfg_path, data_dir, run_root, args.trigger, args.rank, args.lr, end)

    cmd = f"cd /tmp/ai-toolkit && python run.py {cfg_path}"
    sh(cmd)

    latest_path, latest_step = latest_ckpt(run_root)
    summary = {
        "start_step": start,
        "end_step": end,
        "latest_checkpoint": latest_path,
        "latest_step": latest_step,
    }
    Path("/kaggle/working/output/run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

    if eval_after_train and end >= args.total_steps:
        run_checkpoint_eval(ckpt_output_dir, args.trigger)
        run_linkedin_pack(ckpt_output_dir, args.trigger)


if __name__ == "__main__":
    main()
