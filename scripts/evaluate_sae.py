# FILE: clip-sae-interpret_clean/scripts/evaluate_sae.py
# -*- coding: utf-8 -*-

"""
evaluate_sae.py

ZERO-SHOT оценка качества:

1) Базовый CLIP
2) CLIP + Sparse Autoencoder (SAE)

Датасеты:
- CIFAR-10
- Food-101 (subset)

Метрики автоматически сохраняются в JSON и CSV.
"""

import argparse
import os
import json
import csv
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from sae_clip.models.clip_wrapper import CLIPWrapper
from sae_clip.models.sae import SparseAutoencoder
from sae_clip.data.datasets import (
    CIFAR10ZeroShotDataset,
    Food101ZeroShotDataset,
)

# =========================================================
# COLLATE FUNCTION (PIL.Image)
# =========================================================

def pil_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels


# =========================================================
# ZERO-SHOT ACCURACY
# =========================================================

@torch.no_grad()
def zero_shot_accuracy(
    clip_model,
    dataset,
    classnames,
    sae=None,
    batch_size=256,
):
    device = clip_model.device

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pil_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    prompts = [f"a photo of {c.replace('_', ' ')}" for c in classnames]
    
    # Здесь encode_text уже возвращает нормализованные эмбеддинги
    text_emb = clip_model.encode_text(prompts).to(device)
    # УБИРАЕМ ДУБЛИРУЮЩУЮ НОРМАЛИЗАЦИЮ
    # text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Zero-shot eval"):
        labels = labels.to(device)

        # Здесь encode_images уже возвращает нормализованные эмбеддинги
        img_emb = clip_model.encode_images(images)
        # УБИРАЕМ ДУБЛИРУЮЩУЮ НОРМАЛИЗАЦИЮ
        # img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        if sae is not None:
            img_emb, _ = sae(img_emb)
            # Нормализуем после SAE
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        logits = img_emb @ text_emb.T
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# =========================================================
# METRICS SAVE UTILS
# =========================================================

def save_metrics(results, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "zero_shot_metrics.json")
    csv_path = os.path.join(save_dir, "zero_shot_metrics.csv")

    # JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "accuracy", "sae_checkpoint"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[METRICS SAVED]")
    print(f"JSON → {json_path}")
    print(f"CSV  → {csv_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--sae_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    # -----------------------------------------------------
    # CLIP
    # -----------------------------------------------------
    print("[1] Загружаем CLIP...")
    clip_model = CLIPWrapper(device=device)
    clip_model.eval()

    # -----------------------------------------------------
    # SAE (optional)
    # -----------------------------------------------------
    sae = None
    sae_name = None
    if args.sae_path:
        print("[2] Загружаем SAE...")
        sae = SparseAutoencoder()
        sae.load_state_dict(torch.load(args.sae_path, map_location=device))
        sae.to(device)
        sae.eval()
        sae_name = os.path.basename(args.sae_path)

    # =====================================================
    # CIFAR-10
    # =====================================================
    print("\n=== CIFAR-10 ===")

    cifar_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    cifar = CIFAR10ZeroShotDataset(
        root=f"{args.data_root}/cifar-10-batches-py",
        train=False,
    )

    acc = zero_shot_accuracy(
        clip_model, cifar, cifar_classes,
        sae=None, batch_size=args.batch_size
    )
    print(f"CLIP accuracy:        {acc:.4f}")

    results.append({
        "dataset": "CIFAR-10",
        "model": "CLIP",
        "accuracy": acc,
        "sae_checkpoint": None,
    })

    if sae:
        acc = zero_shot_accuracy(
            clip_model, cifar, cifar_classes,
            sae=sae, batch_size=args.batch_size
        )
        print(f"CLIP + SAE accuracy:  {acc:.4f}")

        results.append({
            "dataset": "CIFAR-10",
            "model": "CLIP+SAE",
            "accuracy": acc,
            "sae_checkpoint": sae_name,
        })

    # =====================================================
    # FOOD-101 (subset)
    # =====================================================
    print("\n=== Food-101 (subset=2000) ===")

    food_full = Food101ZeroShotDataset(
        root=f"{args.data_root}",
        split="test",
    )

    food = Subset(food_full, range(2000))
    food_classes = food_full.classes

    acc = zero_shot_accuracy(
        clip_model, food, food_classes,
        sae=None, batch_size=args.batch_size
    )
    print(f"CLIP accuracy:        {acc:.4f}")

    results.append({
        "dataset": "Food-101 (subset=2000)",
        "model": "CLIP",
        "accuracy": acc,
        "sae_checkpoint": None,
    })

    if sae:
        acc = zero_shot_accuracy(
            clip_model, food, food_classes,
            sae=sae, batch_size=args.batch_size
        )
        print(f"CLIP + SAE accuracy:  {acc:.4f}")

        results.append({
            "dataset": "Food-101 (subset=2000)",
            "model": "CLIP+SAE",
            "accuracy": acc,
            "sae_checkpoint": sae_name,
        })

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------
    save_metrics(results)


if __name__ == "__main__":
    main()