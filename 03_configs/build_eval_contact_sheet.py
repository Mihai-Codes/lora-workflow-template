#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(
    "/Users/mihai/mihai-lora-v2/07_kaggle/_kaggle_output_v18/output/mihai_lora_v2/eval"
)
OUT = Path("/Users/mihai/mihai-lora-v2/06_exports/eval_contact_sheet_v18.png")

CHECKPOINTS = [
    "mihai_lora_v2_000001200",
    "mihai_lora_v2_000001400",
    "mihai_lora_v2_000001500",
    "mihai_lora_v2",
]

PROMPTS = [
    "p1_seed43.png",
    "p2_seed44.png",
    "p3_seed45.png",
]


def main() -> None:
    sample = Image.open(ROOT / CHECKPOINTS[0] / PROMPTS[0]).convert("RGB")
    w, h = sample.size
    pad = 20
    label_h = 50
    grid_w = len(CHECKPOINTS) * w + (len(CHECKPOINTS) + 1) * pad
    grid_h = len(PROMPTS) * h + (len(PROMPTS) + 1) * pad + label_h

    canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    # Column labels
    for col, ck in enumerate(CHECKPOINTS):
        x = pad + col * (w + pad)
        draw.text((x, 10), ck, fill=(230, 230, 230))

    for row, prompt in enumerate(PROMPTS):
        y = label_h + pad + row * (h + pad)
        for col, ck in enumerate(CHECKPOINTS):
            x = pad + col * (w + pad)
            img = Image.open(ROOT / ck / prompt).convert("RGB")
            canvas.paste(img, (x, y))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
