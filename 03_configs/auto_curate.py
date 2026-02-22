#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


RAW_DIR = Path("/Users/mihai/mihai-lora-v2/00_raw")
CURATED_DIR = Path("/Users/mihai/mihai-lora-v2/01_curated")
SELECTION_CSV = CURATED_DIR / "selection.csv"

TARGET_COUNT = 36
MIN_SIDE = 720
MAX_DHASH_DISTANCE = 6


@dataclass
class Item:
    path: Path
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    score: float
    dhash: int


def list_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def compute_dhash(gray: np.ndarray) -> int:
    img = Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    bits = arr[:, 1:] > arr[:, :-1]
    out = 0
    for b in bits.flatten():
        out = (out << 1) | int(bool(b))
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]
    return float(np.var(lap))


def image_metrics(path: Path) -> Item | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < MIN_SIDE:
                return None
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
            sharp = laplacian_variance(gray)
            bright = float(np.mean(gray))
            contrast = float(np.std(gray))
            dh = compute_dhash(gray)
    except Exception:
        return None

    return Item(
        path=path,
        width=w,
        height=h,
        sharpness=sharp,
        brightness=bright,
        contrast=contrast,
        score=0.0,
        dhash=dh,
    )


def robust_norm(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 0:
        return vals
    p10 = np.percentile(vals, 10)
    p90 = np.percentile(vals, 90)
    denom = max(1e-9, p90 - p10)
    x = (vals - p10) / denom
    return np.clip(x, 0.0, 1.0)


def exposure_penalty(brightness: float) -> float:
    center = 118.0
    spread = 42.0
    z = (brightness - center) / spread
    return float(math.exp(-(z * z)))


def curate(items: list[Item], target: int) -> tuple[list[Item], set[Path]]:
    if not items:
        return [], set()

    sharp = robust_norm(np.array([i.sharpness for i in items], dtype=np.float32))
    contrast = robust_norm(np.array([i.contrast for i in items], dtype=np.float32))

    for idx, i in enumerate(items):
        exp = exposure_penalty(i.brightness)
        i.score = float(0.6 * sharp[idx] + 0.25 * contrast[idx] + 0.15 * exp)

    ranked = sorted(items, key=lambda x: x.score, reverse=True)
    keep: list[Item] = []
    rejected: set[Path] = set()

    for cand in ranked:
        too_close = any(
            hamming(cand.dhash, chosen.dhash) <= MAX_DHASH_DISTANCE for chosen in keep
        )
        if too_close:
            rejected.add(cand.path)
            continue
        keep.append(cand)
        if len(keep) >= target:
            break

    # If dedupe was too strict and we have fewer than target, backfill by score.
    if len(keep) < target:
        for cand in ranked:
            if cand in keep:
                continue
            keep.append(cand)
            if len(keep) >= target:
                break

    return keep, rejected


def clear_curated_folder(curated_dir: Path) -> None:
    for p in curated_dir.iterdir():
        if p.is_file() and p.name not in {
            ".gitkeep",
            "curation-checklist.md",
            "selection.csv",
        }:
            p.unlink()


def main() -> None:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    imgs = list_images(RAW_DIR)
    items = [m for m in (image_metrics(p) for p in imgs) if m is not None]

    keep, rejected_hash = curate(items, TARGET_COUNT)
    keep_paths = {k.path for k in keep}

    clear_curated_folder(CURATED_DIR)
    for k in keep:
        shutil.copy2(k.path, CURATED_DIR / k.path.name)

    with SELECTION_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filename",
                "keep",
                "reason",
                "score",
                "sharpness",
                "brightness",
                "contrast",
                "width",
                "height",
            ]
        )
        for it in sorted(items, key=lambda x: x.path.name):
            if it.path in keep_paths:
                reason = "selected_by_score"
                keep_flag = "yes"
            elif it.path in rejected_hash:
                reason = "near_duplicate"
                keep_flag = "no"
            else:
                reason = "below_cutoff"
                keep_flag = "no"

            w.writerow(
                [
                    it.path.name,
                    keep_flag,
                    reason,
                    f"{it.score:.4f}",
                    f"{it.sharpness:.2f}",
                    f"{it.brightness:.2f}",
                    f"{it.contrast:.2f}",
                    it.width,
                    it.height,
                ]
            )

    print(f"raw_images={len(imgs)}")
    print(f"usable_images={len(items)}")
    print(f"curated_selected={len(keep)}")
    print(f"selection_csv={SELECTION_CSV}")


if __name__ == "__main__":
    main()
