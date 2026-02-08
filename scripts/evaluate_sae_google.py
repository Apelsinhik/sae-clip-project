# FILE: clip-sae-interpret_clean/scripts/evaluate_sae_google.py
# -*- coding: utf-8 -*-

"""
Упрощенная оценка SAE.
Все модели загружаются с Google Drive.
"""

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

def pil_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels

@torch.no_grad()
def zero_shot_accuracy(clip_model, dataset, classnames, sae=None, batch_size=256):
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

    for images, labels in tqdm(loader, desc="Zero-shot eval"):
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
    json_path = os.path.join(save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # CSV
    csv_path = os.path.join(save_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "accuracy", "notes"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

def main():
    print("="*60)
    print("SAE EVALUATION (Google Drive Models)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    # 1. Загружаем CLIP
    print("\n[1] Loading CLIP...")
    clip_model = CLIPWrapper(device=device)
    clip_model.eval()
    
    # ==================== CIFAR-10 ====================
    print("\n" + "="*40)
    print("CIFAR-10 EVALUATION")
    print("="*40)
    
    cifar_classes = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    
    cifar = CIFAR10ZeroShotDataset(
        root="/content/yadisk/SAE_PROJECT/datasets/cifar-10-batches-py",
        train=False,
    )
    
    # Базовый CLIP
    print("\n[1/2] Testing baseline CLIP...")
    acc_clip = zero_shot_accuracy(clip_model, cifar, cifar_classes, sae=None, batch_size=64)
    print(f"  CLIP accuracy: {acc_clip:.4f}")
    results.append({
        "dataset": "CIFAR-10",
        "model": "CLIP",
        "accuracy": acc_clip,
        "notes": "baseline"
    })
    
    # CLIP + SAE (CIFAR-10)
    sae_cifar_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_cifar_512d/sae_epoch_50.pt"
    if os.path.exists(sae_cifar_path):
        print("\n[2/2] Testing CLIP + SAE (CIFAR-10)...")
        
        sae = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae.load_state_dict(torch.load(sae_cifar_path, map_location=device))
        sae.to(device)
        sae.eval()
        
        acc_sae = zero_shot_accuracy(clip_model, cifar, cifar_classes, sae=sae, batch_size=64)
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
            "notes": "SAE trained on CIFAR-10"
        })
    else:
        print(f"\n⚠️  SAE not found: {sae_cifar_path}")
        print("   Train SAE first: python scripts/train_sae_google_only.py --dataset cifar10")
    
    # ==================== FOOD-101 ====================
    print("\n" + "="*40)
    print("FOOD-101 EVALUATION (subset)")
    print("="*40)
    
    food_full = Food101ZeroShotDataset(
        root="/content/yadisk/SAE_PROJECT/datasets",
        split="test",
    )
    food = Subset(food_full, range(2000))
    food_classes = food_full.classes
    
    # Базовый CLIP
    print("\n[1/2] Testing baseline CLIP...")
    acc_clip_food = zero_shot_accuracy(clip_model, food, food_classes, sae=None, batch_size=64)
    print(f"  CLIP accuracy: {acc_clip_food:.4f}")
    results.append({
        "dataset": "Food-101",
        "model": "CLIP",
        "accuracy": acc_clip_food,
        "notes": "baseline (2000 samples)"
    })
    
    # CLIP + SAE (Food-101)
    sae_food_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_food101_512d/sae_epoch_50.pt"
    if os.path.exists(sae_food_path):
        print("\n[2/2] Testing CLIP + SAE (Food-101)...")
        
        sae = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae.load_state_dict(torch.load(sae_food_path, map_location=device))
        sae.to(device)
        sae.eval()
        
        acc_sae_food = zero_shot_accuracy(clip_model, food, food_classes, sae=sae, batch_size=64)
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
            "notes": "SAE trained on Food-101 (2000 samples)"
        })
    else:
        print(f"\n⚠️  SAE not found: {sae_food_path}")
        print("   Train SAE first: python scripts/train_sae_google_only.py --dataset food101")
    
    # Сохраняем результаты
    save_dir = "/content/drive/MyDrive/SAE_PROJECT/results"
    save_results(results, save_dir)
    
    # Итог
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    for res in results:
        print(f"{res['dataset']:20} | {res['model']:12} | {res['accuracy']:.4f} | {res['notes']}")

if __name__ == "__main__":
    main()

