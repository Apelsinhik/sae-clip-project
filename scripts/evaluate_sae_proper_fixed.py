# FILE: clip-sae-interpret_clean/scripts/evaluate_sae_proper_fixed.py
# -*- coding: utf-8 -*-

"""
ПРАКТИЧЕСКАЯ оценка: только in-domain тесты
CLIP vs CLIP+SAE (где SAE обучен на том же домене)
ВСЕ МОДЕЛИ С GOOGLE DRIVE
"""

import argparse
import os
import json
import csv
import torch
import torch.nn as nn
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
        pin_memory=True if torch.cuda.is_available() else False,
    )

    prompts = [f"a photo of {c.replace('_', ' ')}" for c in classnames]
    text_emb = clip_model.encode_text(prompts).to(device)
    
    print(f"[DEBUG] text_emb shape: {text_emb.shape}")
    print(f"[DEBUG] Expected dimension: 512D")

    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Zero-shot eval"):
        labels = labels.to(device)
        img_emb = clip_model.encode_images(images)
        
        print(f"[DEBUG] img_emb shape: {img_emb.shape}")
        
        if sae is not None:
            # Проверяем совместимость размерностей
            if img_emb.shape[1] != sae.input_dim:
                print(f"❌ ОШИБКА: Несоответствие размерностей!")
                print(f"   img_emb: {img_emb.shape[1]}D, SAE ожидает: {sae.input_dim}D")
                if img_emb.shape[1] == 512 and sae.input_dim == 768:
                    print("   ⚠️  SAE обучен на 768D, но CLIP выдает 512D")
                    print("   ⚠️  Нужно переобучить SAE на 512D!")
                return 0.0
            
            img_emb, _ = sae(img_emb)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        logits = img_emb @ text_emb.T
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0

def save_metrics(results, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "zero_shot_metrics_in_domain.json")
    csv_path = os.path.join(save_dir, "zero_shot_metrics_in_domain.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "accuracy", "sae_checkpoint", "sae_trained_on", "notes"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[IN-DOMAIN METRICS SAVED TO GOOGLE DRIVE]")
    print(f"JSON → {json_path}")
    print(f"CSV  → {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                       help="Путь к датасетам на Яндекс Диске")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--results_dir", type=str, 
                       default="/content/drive/MyDrive/SAE_PROJECT/results",
                       help="Куда сохранить результаты (Google Drive)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    # Загружаем CLIP
    print("[1] Загружаем CLIP-base (512D эмбеддинги)...")
    clip_model = CLIPWrapper(device=device)
    clip_model.eval()

    # ==================== CIFAR-10 ====================
    print("\n" + "="*50)
    print("CIFAR-10 EVALUATION (with CIFAR-10 SAE)")
    print("="*50)

    cifar_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    cifar = CIFAR10ZeroShotDataset(
        root=f"{args.data_root}/cifar-10-batches-py",
        train=False,
    )

    # 1. Базовый CLIP
    print("\n[1/2] Тестируем базовый CLIP...")
    acc_clip = zero_shot_accuracy(
        clip_model, cifar, cifar_classes,
        sae=None, batch_size=args.batch_size
    )
    print(f"📊 CLIP (baseline):           {acc_clip:.4f}")
    results.append({
        "dataset": "CIFAR-10",
        "model": "CLIP",
        "accuracy": acc_clip,
        "sae_checkpoint": None,
        "sae_trained_on": None,
        "notes": "baseline"
    })

    # 2. CLIP + SAE (CIFAR-10 trained) - НОВЫЙ 512D
    sae_cifar_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_cifar_512d/sae_epoch_50.pt"
    if os.path.exists(sae_cifar_path):
        print(f"\n[2/2] Загружаем SAE для CIFAR-10 (512D)...")
        
        # Загружаем SAE с правильной размерностью
        sae_cifar = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae_cifar.load_state_dict(torch.load(sae_cifar_path, map_location=device))
        sae_cifar.to(device)
        sae_cifar.eval()
        
        print(f"[INFO] SAE input_dim: {sae_cifar.input_dim}D (совместимо с CLIP 512D)")

        acc_cifar_sae = zero_shot_accuracy(
            clip_model, cifar, cifar_classes,
            sae=sae_cifar, batch_size=args.batch_size
        )
        print(f"📊 CLIP + SAE(CIFAR-trained): {acc_cifar_sae:.4f}")
        
        diff = acc_cifar_sae - acc_clip
        if diff > 0:
            print(f"   ✅ Улучшение: +{diff:.4f} (+{diff/acc_clip*100:.1f}%)")
        else:
            print(f"   ⚠️  Ухудшение: {diff:.4f} ({diff/acc_clip*100:.1f}%)")

        results.append({
            "dataset": "CIFAR-10",
            "model": "CLIP+SAE",
            "accuracy": acc_cifar_sae,
            "sae_checkpoint": "sae_epoch_50.pt",
            "sae_trained_on": "CIFAR-10",
            "notes": "SAE 512D (новый)"
        })
    else:
        print(f"⚠️  SAE для CIFAR-10 (512D) не найден: {sae_cifar_path}")
        print(f"   Сначала обучите SAE: python scripts/train_sae.py --config configs/train_cifar10_512d.yaml")

    # ==================== FOOD-101 ====================
    print("\n" + "="*50)
    print("FOOD-101 EVALUATION (subset=2000)")
    print("="*50)

    food_full = Food101ZeroShotDataset(root=f"{args.data_root}", split="test")
    food = Subset(food_full, range(2000))
    food_classes = food_full.classes

    # 1. Базовый CLIP
    print("\n[1/2] Тестируем базовый CLIP...")
    acc_clip_food = zero_shot_accuracy(
        clip_model, food, food_classes,
        sae=None, batch_size=args.batch_size
    )
    print(f"📊 CLIP (baseline):           {acc_clip_food:.4f}")
    results.append({
        "dataset": "Food-101",
        "model": "CLIP",
        "accuracy": acc_clip_food,
        "sae_checkpoint": None,
        "sae_trained_on": None,
        "notes": "baseline (2000 samples)"
    })

    # 2. CLIP + SAE (Food-101 trained) - НОВЫЙ 512D
    sae_food_path = "/content/drive/MyDrive/SAE_PROJECT/models/sae_food101_512d/sae_epoch_50.pt"
    if os.path.exists(sae_food_path):
        print(f"\n[2/2] Загружаем SAE для Food-101 (512D)...")
        
        # Загружаем SAE с правильной размерностью
        sae_food = SparseAutoencoder(input_dim=512, latent_dim=4096)
        sae_food.load_state_dict(torch.load(sae_food_path, map_location=device))
        sae_food.to(device)
        sae_food.eval()

        acc_food_sae = zero_shot_accuracy(
            clip_model, food, food_classes,
            sae=sae_food, batch_size=args.batch_size
        )
        print(f"📊 CLIP + SAE(Food-trained):  {acc_food_sae:.4f}")
        
        diff = acc_food_sae - acc_clip_food
        if diff > 0:
            print(f"   ✅ Улучшение: +{diff:.4f} (+{diff/acc_clip_food*100:.1f}%)")
        else:
            print(f"   ⚠️  Ухудшение: {diff:.4f} ({diff/acc_clip_food*100:.1f}%)")

        results.append({
            "dataset": "Food-101",
            "model": "CLIP+SAE",
            "accuracy": acc_food_sae,
            "sae_checkpoint": "sae_epoch_50.pt",
            "sae_trained_on": "Food-101",
            "notes": "SAE 512D (новый, 2000 samples)"
        })
    else:
        print(f"⚠️  SAE для Food-101 (512D) не найден: {sae_food_path}")
        print(f"   Сначала обучите SAE: python scripts/train_sae.py --config configs/train_food101_512d.yaml")

    # Сохраняем результаты на Google Drive
    save_metrics(results, save_dir=args.results_dir)
    
    print("\n" + "="*50)
    print("РЕКОМЕНДАЦИЯ:")
    print("="*50)
    print("Для получения результатов сначала:")
    print("1. Извлеките активации 512D (скрипты extract_clip_activations_*)")
    print("2. Обучите SAE 512D (train_sae.py с configs/train_*_512d.yaml)")
    print("3. Запустите оценку снова")

if __name__ == "__main__":
    main()