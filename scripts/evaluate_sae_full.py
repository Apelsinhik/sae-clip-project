# FILE: clip-sae-interpret_clean/scripts/evaluate_sae_full.py
# -*- coding: utf-8 -*-

"""
Оценка SAE на большем подмножестве Food-101 (10,000 samples)
"""

import os
import json
import csv
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from sae_clip.models.clip_wrapper import CLIPWrapper
from sae_clip.models.sae import SparseAutoencoder
from sae_clip.data.datasets import (
    CIFAR10ZeroShotDataset,
    Food101ZeroShotDataset,
)

def pil_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels

@torch.no_grad()
def zero_shot_accuracy(clip_model, dataset, classnames, sae=None, batch_size=256, desc="Evaluation"):
    device = clip_model.device
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pil_collate_fn,
        num_workers=2,
    )

    prompts = [f"a photo of {c.replace('_', ' ')}" for c in classnames]
    text_emb = clip_model.encode_text(prompts).to(device)

    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc):
        labels = labels.to(device)
        img_emb = clip_model.encode_images(images)

        if sae is not None:
            img_emb, _ = sae(img_emb)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        logits = img_emb @ text_emb.T
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0

def save_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(save_dir, "results_full.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # CSV
    csv_path = os.path.join(save_dir, "results_full.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "model", "accuracy", "samples", "notes"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to datasets on Yandex Disk")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--food_samples", type=int, default=10000,
                       help="How many Food-101 samples to evaluate on")
    parser.add_argument("--results_dir", type=str, 
                       default="/content/drive/MyDrive/SAE_PROJECT/results",
                       help="Where to save results (Google Drive)")
    args = parser.parse_args()

    print("="*70)
    print(f"SAE EVALUATION (Food-101: {args.food_samples:,} samples)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    # 1. Загружаем CLIP
    print("\n[1] Loading CLIP...")
    clip_model = CLIPWrapper(device=device)
    clip_model.eval()
    
    # ==================== CIFAR-10 ====================
    print("\n" + "="*40)
    print("CIFAR-10 EVALUATION (Full test set)")
    print("="*40)
    
    cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    
    cifar = CIFAR10ZeroShotDataset(
        root=f"{args.data_root}/cifar-10-batches-py",
        train=False,
    )
    
    # Базовый CLIP
    print(f"\n[1/2] Testing baseline CLIP...")
    acc_clip = zero_shot_accuracy(
        clip_model, cifar, cifar_classes, 
        sae=None, 
        batch_size=args.batch_size,
        desc="CIFAR-10 CLIP"
    )
    print(f"  CLIP accuracy: {acc_clip:.4f} (10,000 samples)")
    results.append({
        "dataset": "CIFAR-10",
        "model": "CLIP",
        "accuracy": acc_clip,
        "samples": 10000,
        "notes": "full test set"
    })
    
    # CLIP + SAE (CIFAR-10)
    sae_cifar_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_cifar_512d/sae_epoch_50.pt"
    if os.path.exists(sae_cifar_path):
        print(f"\n[2/2] Testing CLIP + SAE (CIFAR-10)...")
        
        sae = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae.load_state_dict(torch.load(sae_cifar_path, map_location=device))
        sae.to(device)
        sae.eval()
        
        acc_sae = zero_shot_accuracy(
            clip_model, cifar, cifar_classes,
            sae=sae,
            batch_size=args.batch_size,
            desc="CIFAR-10 CLIP+SAE"
        )
        print(f"  CLIP+SAE accuracy: {acc_sae:.4f}")
        
        diff = acc_sae - acc_clip
        if diff > 0:
            print(f"  ✅ Improvement: +{diff:.4f} (+{diff/acc_clip*100:.1f}%)")
        else:
            print(f"  ⚠️  Degradation: {diff:.4f} ({diff/acc_clip*100:.1f}%)")
        
        results.append({
            "dataset": "CIFAR-10",
            "model": "CLIP+SAE",
            "accuracy": acc_sae,
            "samples": 10000,
            "notes": "SAE trained on CIFAR-10"
        })
    else:
        print(f"\n⚠️  SAE not found: {sae_cifar_path}")
    
    # ==================== FOOD-101 ====================
    print("\n" + "="*40)
    print(f"FOOD-101 EVALUATION ({args.food_samples:,} samples)")
    print("="*40)
    
    # Загружаем полный датасет
    food_full = Food101ZeroShotDataset(
        root=args.data_root,
        split="test",
    )
    food_classes = food_full.classes
    
    print(f"Full Food-101 test set: {len(food_full):,} images")
    print(f"Classes: {len(food_classes)}")
    
    # Выбираем подмножество (случайно, но стратифицированно)
    if args.food_samples < len(food_full):
        print(f"\nSelecting {args.food_samples:,} random samples...")
        
        # Получаем метки для стратификации
        all_indices = list(range(len(food_full)))
        all_labels = [food_full[i][1] for i in all_indices]
        
        # Стратифицированная выборка
        _, eval_indices = train_test_split(
            all_indices,
            test_size=args.food_samples,
            stratify=all_labels,
            random_state=42
        )
        
        food = Subset(food_full, eval_indices)
        print(f"Selected {len(food):,} samples (stratified)")
    else:
        food = food_full
        print(f"Using full test set: {len(food):,} samples")
    
    # Базовый CLIP
    print(f"\n[1/2] Testing baseline CLIP...")
    acc_clip_food = zero_shot_accuracy(
        clip_model, food, food_classes,
        sae=None,
        batch_size=args.batch_size,
        desc="Food-101 CLIP"
    )
    print(f"  CLIP accuracy: {acc_clip_food:.4f} ({len(food):,} samples)")
    results.append({
        "dataset": "Food-101",
        "model": "CLIP",
        "accuracy": acc_clip_food,
        "samples": len(food),
        "notes": f"{len(food):,} samples"
    })
    
    # CLIP + SAE (Food-101)
    sae_food_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_food101_512d/sae_epoch_50.pt"
    if os.path.exists(sae_food_path):
        print(f"\n[2/2] Testing CLIP + SAE (Food-101)...")
        
        sae = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae.load_state_dict(torch.load(sae_food_path, map_location=device))
        sae.to(device)
        sae.eval()

        acc_sae_food = zero_shot_accuracy(
            clip_model, food, food_classes,
            sae=sae,
            batch_size=args.batch_size,
            desc="Food-101 CLIP+SAE"
        )
        print(f"  CLIP+SAE accuracy: {acc_sae_food:.4f}")
        
        diff = acc_sae_food - acc_clip_food
        if diff > 0:
            print(f"  ✅ Improvement: +{diff:.4f} (+{diff/acc_clip_food*100:.1f}%)")
        else:
            print(f"  ⚠️  Degradation: {diff:.4f} ({diff/acc_clip_food*100:.1f}%)")
        
        results.append({
            "dataset": "Food-101",
            "model": "CLIP+SAE",
            "accuracy": acc_sae_food,
            "samples": len(food),
            "notes": f"SAE trained on Food-101 ({len(food):,} samples)"
        })
    else:
        print(f"\n⚠️  SAE not found: {sae_food_path}")
        print("   Train SAE first: python scripts/train_sae_google_only.py --dataset food101")
    
    # Сохраняем результаты
    save_results(results, args.results_dir)
    
    # Итог
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    for res in results:
        print(f"{res['dataset']:20} | {res['model']:12} | {res['accuracy']:.4f} | "
              f"{res['samples']:6,} samples | {res['notes']}")
    
    # Статистический анализ
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS:")
    print("="*70)
    
    # Для Food-101
    if len(results) >= 4:
        food_clip_acc = results[2]['accuracy']
        food_sae_acc = results[3]['accuracy']
        food_samples = results[2]['samples']
        
        # Стандартная ошибка
        se = np.sqrt((food_clip_acc * (1 - food_clip_acc)) / food_samples)
        
        print(f"Food-101 Results:")
        print(f"  CLIP accuracy: {food_clip_acc:.4f}")
        print(f"  CLIP+SAE accuracy: {food_sae_acc:.4f}")
        print(f"  Improvement: +{food_sae_acc - food_clip_acc:.4f}")
        print(f"  Relative improvement: +{(food_sae_acc/food_clip_acc - 1)*100:.1f}%")
        print(f"  Standard error (CLIP): ±{1.96*se:.4f} (95% CI)")
        
        # Z-тест для пропорций
        p1, p2 = food_clip_acc, food_sae_acc
        n = food_samples
        p_pool = (p1 + p2) / 2
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (2/n))
        z_score = (p2 - p1) / se_pool
        
        print(f"  Z-score: {z_score:.2f}")
        if abs(z_score) > 1.96:
            print(f"  ✅ Statistically significant at p < 0.05")
        else:
            print(f"  ⚠️  Not statistically significant at p < 0.05")

if __name__ == "__main__":
    main()