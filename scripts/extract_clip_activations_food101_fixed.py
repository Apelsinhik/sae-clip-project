# FILE: clip-sae-interpret_clean/scripts/extract_clip_activations_food101_fixed.py
# -*- coding: utf-8 -*-

"""
Исправленный скрипт для генерации CLIP-активаций для Food-101 (512D)

Использует transformers вместо clip library
СОХРАНЕНИЕ НА GOOGLE DRIVE
"""

import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_dataset_path(relative_path):
    """
    Универсальный поиск папки с данными:
    сначала проверяем Яндекс Диск,
    затем Google Drive
    """

    yandex_base = "/content/yadisk/SAE_PROJECT"
    google_base = "/content/drive/MyDrive/SAE_PROJECT"

    yandex_path = os.path.join(yandex_base, relative_path)
    google_path = os.path.join(google_base, relative_path)

    if os.path.exists(yandex_path):
        print(f"[INFO] Используем данные с Яндекс Диска: {yandex_path}")
        return yandex_path

    if os.path.exists(google_path):
        print(f"[INFO] Используем данные с Google Drive: {google_path}")
        return google_path

    raise FileNotFoundError(
        f"Датасет не найден ни на Яндекс Диске, ни на Google Drive: {relative_path}"
    )


def collect_image_paths(root_dir, max_images=-1):
    """
    Сбор путей ко всем изображениям в папке датасета.
    При необходимости ограничивает число изображений.
    """

    paths = []

    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, fname))

                if max_images > 0 and len(paths) >= max_images:
                    return paths

    return paths


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # ARGUMENT PARSING
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Генерация CLIP-активаций для Food-101 (512D) - Google Drive"
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Относительный путь к папке с изображениями Food-101",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Куда сохранить файл с активациями (.pt) НА GOOGLE DRIVE",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="Сколько изображений обработать (-1 = все)",
    )

    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP модель для извлечения эмбеддингов",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Размер батча для обработки",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Устройство: {device}")
    print(f"[INFO] CLIP модель: {args.clip_model}")
    print(f"[INFO] Данные читаем с Яндекс Диска")
    print(f"[INFO] Результат сохраняем на Google Drive: {args.output_path}")

    # Проверяем, что выходной путь на Google Drive
    if not args.output_path.startswith("/content/drive/MyDrive"):
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Выходной путь не на Google Drive!")
        print(f"   Рекомендуется: /content/drive/MyDrive/SAE_PROJECT/activations/...")

    # -----------------------------------------------------
    # FIND DATASET
    # -----------------------------------------------------
    print("\n[1] Поиск папки с датасетом")
    images_root = find_dataset_path(args.images_dir)

    # -----------------------------------------------------
    # COLLECT IMAGES
    # -----------------------------------------------------
    print("\n[2] Сбор путей к изображениям")
    image_paths = collect_image_paths(images_root, args.max_images)
    print(f"Найдено изображений: {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError("Не найдено ни одного изображения для обработки!")

    # -----------------------------------------------------
    # LOAD CLIP (TRANSFORMERS)
    # -----------------------------------------------------
    print(f"\n[3] Загрузка модели CLIP: {args.clip_model}")
    
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()
    
    # Проверяем размерность
    projection_dim = model.config.projection_dim
    print(f"[INFO] Размерность эмбеддингов: {projection_dim}D")
    
    if projection_dim != 512:
        print(f"⚠️  Внимание: размерность {projection_dim}D, ожидается 512D для SAE")

    # -----------------------------------------------------
    # EXTRACT ACTIVATIONS (BATCHED)
    # -----------------------------------------------------
    print(f"\n[4] Генерация эмбеддингов (батчами по {args.batch_size})")
    
    all_activations = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Processing"):
            
            batch_paths = image_paths[i : i + args.batch_size]
            batch_images = []
            
            # Загружаем изображения
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    print(f"[WARNING] Ошибка загрузки {p}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Обработка через CLIP процессор
            inputs = processor(
                images=batch_images, 
                return_tensors="pt",
                padding=True
            ).to(device)
            
            # ИСПРАВЛЕНИЕ: Используем vision_model напрямую
            vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
            
            # Берем CLS токен (первый токен)
            image_features = vision_outputs.last_hidden_state[:, 0, :]
            
            # Проецируем через visual_projection
            image_features = model.visual_projection(image_features)
            
            # Нормализуем
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_activations.append(image_features.cpu())

    # -----------------------------------------------------
    # SAVE RESULT ON GOOGLE DRIVE
    # -----------------------------------------------------
    print("\n[5] Сохранение результата на Google Drive")
    
    if not all_activations:
        raise RuntimeError("Не удалось извлечь ни одного эмбеддинга!")
    
    activations_tensor = torch.cat(all_activations, dim=0)
    
    print(f"    Форма тензора: {activations_tensor.shape}")
    print(f"    Тип данных: {activations_tensor.dtype}")

    # Создаем папку если нужно
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Сохраняем на Google Drive
    torch.save(activations_tensor, args.output_path)

    print("\n[✅] СОХРАНЕНО НА GOOGLE DRIVE")
    print(f"Файл с активациями → {args.output_path}")
    print(f"Размер тензора: {activations_tensor.shape}")
    print("Генерация активаций завершена.")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()