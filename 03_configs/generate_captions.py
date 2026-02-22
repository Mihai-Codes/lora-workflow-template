#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


CURATED_DIR = Path("/Users/mihai/mihai-lora-v2/01_curated")
CAPTIONS_DIR = Path("/Users/mihai/mihai-lora-v2/02_captions")
TRIGGER = "mihai"

BASE_CAPTIONS = [
    "photo of {t}, professional headshot, natural skin texture, soft studio lighting, clean background",
    "photo of {t}, business portrait, realistic lighting, subtle expression, office-style background",
    "photo of {t}, close-up professional portrait, photorealistic, neutral background, high detail",
    "photo of {t}, upper body business headshot, natural skin detail, soft key light, minimal background",
]


def list_curated_images() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    return sorted(
        [p for p in CURATED_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def clear_old_captions() -> None:
    for p in CAPTIONS_DIR.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() == ".txt"
            and p.name != "caption-template.txt"
        ):
            p.unlink()


def main() -> None:
    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    clear_old_captions()
    imgs = list_curated_images()

    for idx, img in enumerate(imgs):
        template = BASE_CAPTIONS[idx % len(BASE_CAPTIONS)]
        caption = template.format(t=TRIGGER)
        out = CAPTIONS_DIR / f"{img.stem}.txt"
        out.write_text(caption + "\n", encoding="utf-8")

    print(f"curated_images={len(imgs)}")
    print(f"captions_written={len(imgs)}")


if __name__ == "__main__":
    main()
