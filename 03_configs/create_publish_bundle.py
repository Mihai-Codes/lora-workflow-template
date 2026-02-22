#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path("/Users/mihai/mihai-lora-v2")
BUNDLE = ROOT / "publish_bundle"

FILES = [
    ".gitignore",
    "README-public.md",
    "PUBLISHING.md",
    "README.md",
    "03_configs/auto_curate.py",
    "03_configs/build_eval_contact_sheet.py",
    "03_configs/build_replicate_bundle.py",
    "02_captions/caption-template.txt",
    "05_validation/fixed-prompts.txt",
    "03_configs/colab-free-runbook.md",
    "03_configs/colab_cells_template.py",
    "03_configs/create_publish_bundle.py",
    "03_configs/generate_captions.py",
    "03_configs/kaggle-runbook.md",
    "03_configs/kaggle_cli_workflow.sh",
    "03_configs/monitor_kaggle_run.sh",
    "03_configs/prepare_kaggle_assets.py",
    "03_configs/prepare_kaggle_checkpoints.py",
    "03_configs/replicate-v2-run-plan.md",
    "03_configs/replicate_run_commands.md",
    "07_kaggle/train_flux_lora.py",
    "07_kaggle/kernel-metadata.template.json",
    "07_kaggle/dataset-metadata.template.json",
    "08_kaggle_eval/evaluate_checkpoints.py",
    "08_kaggle_eval/kernel-metadata.json",
]


def copy_file(rel: str) -> None:
    src = ROOT / rel
    if not src.exists():
        return
    dst = BUNDLE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True, exist_ok=True)

    for rel in FILES:
        copy_file(rel)

    print(f"bundle_created={BUNDLE}")
    print(f"files_copied={len([p for p in BUNDLE.rglob('*') if p.is_file()])}")


if __name__ == "__main__":
    main()
