# evaluate_sae_fixed.py
import argparse
import os
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# Импортируем с исправлением
from sae_clip.models.clip_wrapper import CLIPWrapper
from sae_clip.models.sae import SparseAutoencoder
from sae_clip.data.datasets import (
    CIFAR10ZeroShotDataset,
    Food101ZeroShotDataset,
)

# Создаем обертку, которая использует CLIP-base
class CLIPWrapperBase(CLIPWrapper):
    def __init__(self, device: str = None):
        # Принудительно используем base версию
        super().__init__(model_name="openai/clip-vit-base-patch32", device=device)

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
    )

    prompts = [f"a photo of {c.replace('_', ' ')}" for c in classnames]
    text_emb = clip_model.encode_text(prompts).to(device)
    
    # Проверка размерности
    print(f"[DEBUG] text_emb shape: {text_emb.shape}")
    expected_dim = 768 if sae is None else sae.input_dim
    if text_emb.shape[1] != expected_dim:
        print(f"⚠️  Предупреждение: text_emb имеет размерность {text_emb.shape[1]}, ожидается {expected_dim}")

    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Zero-shot eval"):
        labels = labels.to(device)
        img_emb = clip_model.encode_images(images)
        
        print(f"[DEBUG] img_emb shape: {img_emb.shape}")  # Должно быть (batch, 768)

        if sae is not None:
            # Проверяем совместимость
            if img_emb.shape[1] != sae.input_dim:
                print(f"❌ Ошибка: Несоответствие размерностей!")
                print(f"   img_emb: {img_emb.shape[1]}D, SAE ожидает: {sae.input_dim}D")
                return 0.0
            
            img_emb, _ = sae(img_emb)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        logits = img_emb @ text_emb.T
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

def save_metrics(results, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "zero_shot_metrics_fixed.json")
    csv_path = os.path.join(save_dir, "zero_shot_metrics_fixed.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "accuracy", "sae_checkpoint", "sae_trained_on"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[METRICS SAVED]")
    print(f"JSON → {json_path}")
    print(f"CSV  → {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    print("[1] Загружаем CLIP-base (768D)...")
    clip_model = CLIPWrapperBase(device=device)
    clip_model.eval()

    # Проверяем размерность CLIP
    test_emb = clip_model.encode_text(["test"])
    print(f"[INFO] CLIP embedding dimension: {test_emb.shape[1]}D")

    # ==================== CIFAR-10 ====================
    print("\n" + "="*50)
    print("CIFAR-10 EVALUATION")
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
        "sae_trained_on": None
    })

    # 2. CLIP + SAE (CIFAR-10 trained)
    sae_cifar_path = "/content/yadisk/SAE_PROJECT/models/sae_cifar/sae_epoch_50.pt"
    if os.path.exists(sae_cifar_path):
        print(f"\n[2/2] Загружаем и тестируем SAE для CIFAR-10...")
        sae_cifar = SparseAutoencoder(input_dim=768, latent_dim=4096)
        sae_cifar.load_state_dict(torch.load(sae_cifar_path, map_location=device))
        sae_cifar.to(device)
        sae_cifar.eval()
        
        print(f"[INFO] SAE input_dim: {sae_cifar.input_dim}D (совместимо с CLIP)")

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
            "sae_trained_on": "CIFAR-10"
        })
    else:
        print(f"⚠️  SAE для CIFAR-10 не найден: {sae_cifar_path}")

    # ==================== FOOD-101 ====================
    print("\n" + "="*50)
    print("FOOD-101 EVALUATION")
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
        "dataset": "Food-101 (subset=2000)",
        "model": "CLIP",
        "accuracy": acc_clip_food,
        "sae_checkpoint": None,
        "sae_trained_on": None
    })

    # 2. CLIP + SAE (Food-101 trained)
    sae_food_path = "/content/yadisk/SAE_PROJECT/models/sae_food101/sae_epoch_50.pt"
    if os.path.exists(sae_food_path):
        print(f"\n[2/2] Загружаем и тестируем SAE для Food-101...")
        sae_food = SparseAutoencoder(input_dim=768, latent_dim=4096)
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
            "dataset": "Food-101 (subset=2000)",
            "model": "CLIP+SAE",
            "accuracy": acc_food_sae,
            "sae_checkpoint": "sae_epoch_50.pt",
            "sae_trained_on": "Food-101"
        })
    else:
        print(f"⚠️  SAE для Food-101 не найден: {sae_food_path}")

    # Сохраняем результаты
    save_metrics(results)
    
    print("\n" + "="*50)
    print("ОЦЕНКА ЗАВЕРШЕНА!")
    print("="*50)

if __name__ == "__main__":
    main()