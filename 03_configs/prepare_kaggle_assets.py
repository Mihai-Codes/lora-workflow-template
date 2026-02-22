#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path("/Users/mihai/mihai-lora-v2")
KAGGLE_DIR = ROOT / "07_kaggle"
ZIP_SRC = ROOT / "03_configs" / "replicate_bundle_v2.zip"


def fill_template(template_path: Path, out_path: Path, username: str) -> None:
    text = template_path.read_text(encoding="utf-8")
    text = text.replace("__KAGGLE_USERNAME__", username)
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    username = os.getenv("KAGGLE_USERNAME", "")
    if not username:
        raise SystemExit("Set KAGGLE_USERNAME in your environment first.")

    if not ZIP_SRC.exists():
        raise SystemExit(f"Missing zip bundle: {ZIP_SRC}")

    data_dir = KAGGLE_DIR / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(ZIP_SRC, data_dir / "replicate_bundle_v2.zip")

    fill_template(
        KAGGLE_DIR / "dataset-metadata.template.json",
        data_dir / "dataset-metadata.json",
        username,
    )
    fill_template(
        KAGGLE_DIR / "kernel-metadata.template.json",
        KAGGLE_DIR / "kernel-metadata.json",
        username,
    )

    print(f"Prepared Kaggle assets for username={username}")
    print(f"Dataset dir: {data_dir}")
    print(f"Kernel metadata: {KAGGLE_DIR / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()
