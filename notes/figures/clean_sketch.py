#!/usr/bin/env python3
"""
Clean paper background from sketch photos for use in scientific diagrams.

Flattens uneven lighting, whitens the paper, and preserves original pencil
strokes and any pre-existing grid dots.

Usage:
    python clean_sketch_bg.py IMG_4172.jpeg IMG_4173.jpeg ...
    python clean_sketch_bg.py *.jpeg --suffix _clean --blur 60 --threshold 235
"""
import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageFilter


def clean(path: str, blur: float, threshold: int, suffix: str) -> str:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32)

    # Estimate the paper background with a large Gaussian blur, then divide
    # to flatten uneven lighting/shadows.
    bg = np.array(img.filter(ImageFilter.GaussianBlur(radius=blur))).astype(np.float32)
    norm = np.clip(arr / np.maximum(bg, 1) * 255.0, 0, 255)

    # Snap near-white pixels to pure white; keep darker pencil tones intact.
    lum = norm.mean(axis=2)
    norm[lum > threshold] = 255

    base, ext = os.path.splitext(path)
    out_path = f"{base}{suffix}.png"
    Image.fromarray(norm.astype(np.uint8)).save(out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+", help="Input image files")
    p.add_argument("--blur", type=float, default=60, help="Background blur radius (px)")
    p.add_argument("--threshold", type=int, default=235, help="Luminance >= this becomes pure white (0-255)")
    p.add_argument("--suffix", default="_clean", help="Suffix for output filenames")
    args = p.parse_args()

    for path in args.images:
        try:
            out = clean(path, args.blur, args.threshold, args.suffix)
            print(f"{path} -> {out}")
        except Exception as e:
            print(f"{path}: ERROR {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
