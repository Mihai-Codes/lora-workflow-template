#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


ROOT = Path("/Users/mihai/mihai-lora-v2")
KAGGLE_DIR = ROOT / "07_kaggle"
CHECKPOINT_DATASET_DIR = KAGGLE_DIR / "checkpoints_dataset"


def latest_output_dir() -> Path:
    candidates = list(KAGGLE_DIR.glob("_kaggle_output_v*/output/mihai_lora_v2"))
    if not candidates:
        raise SystemExit("No local Kaggle output folders found")

    def version_key(p: Path) -> int:
        name = p.parts[-3]  # _kaggle_output_v14
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    return sorted(candidates, key=version_key)[-1]


def main() -> None:
    username = os.getenv("KAGGLE_USERNAME", "")
    if not username:
        raise SystemExit("Set KAGGLE_USERNAME first")

    latest_out = latest_output_dir()
    if not latest_out.exists():
        raise SystemExit(f"Missing latest output folder: {latest_out}")

    if CHECKPOINT_DATASET_DIR.exists():
        shutil.rmtree(CHECKPOINT_DATASET_DIR)
    CHECKPOINT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in sorted(latest_out.glob("*.safetensors")):
        shutil.copy2(p, CHECKPOINT_DATASET_DIR / p.name)
        copied += 1

    opt = latest_out / "optimizer.pt"
    if opt.exists():
        shutil.copy2(opt, CHECKPOINT_DATASET_DIR / opt.name)

    meta = {
        "id": f"{username}/mihai-lora-v2-checkpoints",
        "title": "Mihai LoRA v2 Checkpoints",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (CHECKPOINT_DATASET_DIR / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    kernel_meta_path = KAGGLE_DIR / "kernel-metadata.json"
    kernel_meta = json.loads(kernel_meta_path.read_text(encoding="utf-8"))
    data_sources = set(kernel_meta.get("dataset_sources", []))
    data_sources.add(f"{username}/mihai-lora-v2-data")
    data_sources.add(f"{username}/mihai-lora-v2-checkpoints")
    kernel_meta["dataset_sources"] = sorted(data_sources)
    kernel_meta_path.write_text(
        json.dumps(kernel_meta, indent=2) + "\n", encoding="utf-8"
    )

    print(f"copied_checkpoints={copied}")
    print(f"source_output={latest_out}")
    print(f"checkpoint_dataset_dir={CHECKPOINT_DATASET_DIR}")
    print(f"kernel_sources={kernel_meta['dataset_sources']}")


if __name__ == "__main__":
    main()
