#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path("/Users/mihai/mihai-lora-v2")
CURATED_DIR = ROOT / "01_curated"
CAPTIONS_DIR = ROOT / "02_captions"
BUNDLE_DIR = ROOT / "03_configs" / "replicate_bundle"
ZIP_BASE = ROOT / "03_configs" / "replicate_bundle_v2"


def list_images() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    return sorted(
        [p for p in CURATED_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def main() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    imgs = list_images()
    copied = 0
    for img in imgs:
        txt = CAPTIONS_DIR / f"{img.stem}.txt"
        if not txt.exists():
            continue
        shutil.copy2(img, BUNDLE_DIR / img.name)
        shutil.copy2(txt, BUNDLE_DIR / txt.name)
        copied += 1

    zip_path = shutil.make_archive(str(ZIP_BASE), "zip", str(BUNDLE_DIR))
    print(f"paired_items={copied}")
    print(f"bundle_dir={BUNDLE_DIR}")
    print(f"zip_file={zip_path}")


if __name__ == "__main__":
    main()
