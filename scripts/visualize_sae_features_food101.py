# FILE: clip-sae-interpret_clean/scripts/visualize_sae_features_food101.py
# -*- coding: utf-8 -*-

"""
visualize_sae_features_food101.py

Визуализация SAE-фич для датасета Food-101.

Назначение:
- построение коллажей для топ-K изображений каждой фичи
- без использования моделей
- без GPU
- только визуальная проверка того, что выучил SAE

Вход:
- CSV с top-K активациями
- папка с изображениями Food-101

Выход:
- папка с коллажами
"""

import os
import argparse
import pandas as pd
from PIL import Image
import math


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def collect_all_images(dataset_root):
    """
    Собираем список всех изображений Food-101
    в стабильном порядке
    """

    paths = []

    for class_name in sorted(os.listdir(dataset_root)):
        class_dir = os.path.join(dataset_root, class_name)

        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, fname))

    return paths


def make_collage(image_paths, grid, save_path):
    """
    Создание простого коллажа из изображений
    """

    images = [Image.open(p).convert("RGB") for p in image_paths]

    size = images[0].size
    w, h = size

    rows = grid
    cols = grid

    collage = Image.new("RGB", (cols * w, rows * h))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols

        collage.paste(img, (c * w, r * h))

    collage.save(save_path)


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Визуализация SAE-фич для Food-101"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="CSV с top-K активациями",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Папка с изображениями Food-101",
    )

    parser.add_argument(
        "--features",
        type=int,
        default=300,
        help="Сколько фич визуализировать",
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=3,
        help="Размер сетки (grid x grid)",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/feature_viz_food101",
        help="Куда сохранять коллажи",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("[1] Loading CSV")

    df = pd.read_csv(args.csv_path)

    print("[2] Collecting image paths")

    all_images = collect_all_images(args.dataset_root)

    print(f"Всего изображений найдено: {len(all_images)}")

    feature_ids = sorted(df["feature_id"].unique())[: args.features]

    print(f"Visualizing features: {feature_ids}")

    print("[3] Building collages")

    for fid in feature_ids:

        sub = df[df["feature_id"] == fid]

        top_indices = sub.sort_values("rank")["image_index"].values

        # Берём первые grid*grid изображений
        k = args.grid * args.grid
        top_indices = top_indices[:k]

        image_paths = [all_images[i] for i in top_indices]

        save_path = os.path.join(args.save_dir, f"feature_{fid:04d}.png")

        make_collage(image_paths, args.grid, save_path)

    print("\n[DONE]")
    print(f"Коллажи сохранены в → {args.save_dir}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
