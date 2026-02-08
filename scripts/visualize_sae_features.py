# FILE: clip-sae-interpret_clean/scripts/visualize_sae_features.py
# -*- coding: utf-8 -*-

"""
visualize_sae_features.py

Визуализация SAE-фич:
- берёт top-K изображений для каждой фичи
- собирает коллажи (grid x grid)
- сохраняет PNG

Работает ТОЛЬКО по индексам — ничего не пересчитывает.
"""

import os
import argparse
import pandas as pd
import math
from PIL import Image
from torchvision.datasets import CIFAR10


# =========================================================
# UTILS
# =========================================================

def make_grid(images, grid_size, upscale=4):
    """
    Собирает список PIL.Image в квадратный коллаж
    и увеличивает его для удобной визуализации.

    images   : list[PIL.Image]
    grid_size: int (например, 4 → 4x4)
    upscale  : во сколько раз увеличить коллаж
    """

    w, h = images[0].size
    grid = Image.new("RGB", (grid_size * w, grid_size * h))

    for idx, img in enumerate(images):
        x = (idx % grid_size) * w
        y = (idx // grid_size) * h
        grid.paste(img, (x, y))

    # ⬆️ УВЕЛИЧЕНИЕ КОЛЛАЖА
    if upscale > 1:
        grid = grid.resize(
            (grid.width * upscale, grid.height * upscale),
            resample=Image.NEAREST  # важно для CIFAR
        )

    return grid


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAE features with CIFAR-10 images"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="CSV с top-K активациями SAE",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Путь к datasets",
    )

    parser.add_argument(
        "--features",
        type=int,
        default=9,
        help="Сколько SAE-фич визуализировать",
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=3,
        help="Размер сетки (3 = 3x3)",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/feature_viz",
        help="Куда сохранять коллажи",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------
    # 1. LOAD CSV
    # -----------------------------------------------------
    print("[1] Loading CSV")
    df = pd.read_csv(args.csv_path)

    feature_ids = sorted(df["feature_id"].unique())[: args.features]
    print(f"Visualizing features: {feature_ids}")

    # -----------------------------------------------------
    # 2. LOAD CIFAR-10
    # -----------------------------------------------------
    print("[2] Loading CIFAR-10")
    cifar = CIFAR10(
        root=args.dataset_root,   # ← КОРЕНЬ datasets
        train=True,
        download=False,
    )

    # -----------------------------------------------------
    # 3. VISUALIZE
    # -----------------------------------------------------
    print("[3] Creating collages")

    images_per_feature = args.grid * args.grid

    for fid in feature_ids:
        rows = df[df["feature_id"] == fid].head(images_per_feature)

        imgs = []
        for _, row in rows.iterrows():
            img, _ = cifar[int(row["image_index"])]
            imgs.append(img)

        collage = make_grid(imgs, args.grid)

        out_path = os.path.join(
            args.save_dir, f"feature_{fid:04d}.png"
        )
        collage.save(out_path)

        print(f"[SAVED] {out_path}")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()
