# FILE: clip-sae-interpret_clean/scripts/extract_clip_activations_cifar10.py
# -*- coding: utf-8 -*-

"""
Извлечение CLIP-активаций для CIFAR-10 (512D эмбеддинги)

Использует CLIP-base (ViT-B/32) для совместимости с SAE 512D
СОХРАНЕНИЕ НА GOOGLE DRIVE
"""

import os
import argparse
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

def pil_collate_fn(batch):
    """Collate function для PIL изображений"""
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Путь к папке с данными CIFAR-10 (Яндекс Диск)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Куда сохранить .pt файл с активациями (Google Drive)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                       help="CLIP модель для извлечения эмбеддингов")
    parser.add_argument("--train_split", action="store_true",
                       help="Использовать train split (50k) вместо test (10k)")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Устройство: {device}")
    print(f"[INFO] CLIP модель: {args.clip_model}")
    print(f"[INFO] Split: {'train' if args.train_split else 'test'}")
    print(f"[INFO] Данные с: {args.data_root} (Яндекс Диск)")
    print(f"[INFO] Сохранение в: {args.output_path} (Google Drive)")

    # Проверяем, что выходной путь на Google Drive
    if not args.output_path.startswith("/content/drive/MyDrive"):
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Выходной путь не на Google Drive!")
        print(f"   Рекомендуется: /content/drive/MyDrive/SAE_PROJECT/activations/...")

    # -----------------------------------------------------
    # ЗАГРУЗКА CLIP
    # -----------------------------------------------------
    print("\n[1] Загрузка CLIP модели...")
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()
    
    # Проверяем размерность
    projection_dim = model.config.projection_dim
    print(f"[INFO] Размерность эмбеддингов: {projection_dim}D")
    if projection_dim != 512:
        print(f"⚠️  Внимание: размерность {projection_dim}D, ожидается 512D для SAE")

    # -----------------------------------------------------
    # ЗАГРУЗКА CIFAR-10
    # -----------------------------------------------------
    print("\n[2] Загрузка CIFAR-10 датасета...")
    dataset = CIFAR10(
        root=args.data_root,
        train=args.train_split,  # True для train, False для test
        download=True
    )
    
    print(f"[INFO] Загружено изображений: {len(dataset)}")
    
    # -----------------------------------------------------
    # DATALOADER
    # -----------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pil_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # -----------------------------------------------------
    # ИЗВЛЕЧЕНИЕ АКТИВАЦИЙ
    # -----------------------------------------------------
    print(f"\n[3] Извлечение CLIP эмбеддингов...")
    all_activations = []
    
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Processing"):
            # Обработка через CLIP процессор
            inputs = processor(
                images=images, 
                return_tensors="pt",
                padding=True
            ).to(device)
            
            # СПОСОБ 1: Используем vision_model напрямую (гарантированный тензор)
            vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
            
            # Берем CLS токен (первый токен)
            image_features = vision_outputs.last_hidden_state[:, 0, :]
            
            # Проецируем через visual_projection
            image_features = model.visual_projection(image_features)
            
            # Нормализуем
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_activations.append(image_features.cpu())
    
    # -----------------------------------------------------
    # СОХРАНЕНИЕ НА GOOGLE DRIVE
    # -----------------------------------------------------
    activations_tensor = torch.cat(all_activations, dim=0)
    
    print(f"\n[4] Сохранение на Google Drive...")
    print(f"    Форма тензора: {activations_tensor.shape}")
    print(f"    Тип данных: {activations_tensor.dtype}")
    
    # Создаем папку если нужно
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Сохраняем на Google Drive
    torch.save(activations_tensor, args.output_path)
    
    print(f"\n[✅] ГОТОВО! Сохранено на Google Drive")
    print(f"    Файл: {args.output_path}")
    print(f"    Размер: {activations_tensor.shape}")
    print(f"    Ожидаемая форма: ({len(dataset)}, {projection_dim})")

if __name__ == "__main__":
    main()