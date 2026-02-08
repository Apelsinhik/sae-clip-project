# FILE: clip-sae-interpret_clean/scripts/train_sae.py
# -*- coding: utf-8 -*-

"""
train_sae.py

CLI-скрипт для обучения Sparse Autoencoder (SAE) на CLIP-активациях.
"""

import os
import argparse
import yaml
import torch
import shutil

from sae_clip.models.sae import SparseAutoencoder
from sae_clip.training.trainer import SAETrainer
from sae_clip.utils.storage import (
    resolve_activations_path,
    resolve_checkpoint,
    resolve_save_dirs,
)

# =========================================================
# CONFIG UTILS
# =========================================================

def load_config(path: str) -> dict:
    """Загружает YAML-конфиг обучения."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Преобразуем строки в числа где нужно
    return convert_strings_to_numbers(cfg)

def convert_strings_to_numbers(obj):
    """Рекурсивно преобразует строки в числа в конфиге."""
    if isinstance(obj, dict):
        return {k: convert_strings_to_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_strings_to_numbers(v) for v in obj]
    elif isinstance(obj, str):
        # Пробуем преобразовать строку в число
        try:
            # Для научной нотации (5e-4, 1e-3 и т.д.)
            if 'e' in obj.lower():
                return float(obj)
            # Для целых чисел
            elif obj.isdigit() or (obj.startswith('-') and obj[1:].isdigit()):
                return int(obj)
            # Для чисел с плавающей точкой
            elif '.' in obj and obj.replace('.', '', 1).replace('-', '', 1).isdigit():
                return float(obj)
            else:
                return obj
        except:
            return obj
    else:
        return obj

# =========================================================
# MAIN
# =========================================================

def main():
    # -----------------------------------------------------
    # 1. ARGPARSE
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoder (SAE) on CLIP activations"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/train_config.yaml",
        help="Путь к YAML-конфигу обучения",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение из checkpoint (если найден)",
    )

    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Игнорировать checkpoint и обучать с нуля",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # 2. LOAD CONFIG
    # -----------------------------------------------------
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== CONFIGURATION ===")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print("=====================\n")

    # -----------------------------------------------------
    # 3. LOAD ACTIVATIONS
    # -----------------------------------------------------
    print("[1] LOAD ACTIVATIONS")

    activations_path = resolve_activations_path(cfg)
    print(f"  Path: {activations_path}")

    # Загружаем активации
    data = torch.load(activations_path, map_location="cpu")
    print(f"  Loaded: {type(data)}")

    # Преобразуем в правильный формат
    if isinstance(data, torch.Tensor):
        print(f"  Tensor shape: {data.shape}")
        data = {"activations": data}
    elif isinstance(data, dict):
        print(f"  Dict keys: {list(data.keys())}")
        if "activations" not in data:
            # Ищем тензор в словаре
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data = {"activations": value}
                    print(f"  Using key '{key}' as activations")
                    break
    
    print(f"  Final shape: {data['activations'].shape}")

    # -----------------------------------------------------
    # 4. INIT SAE
    # -----------------------------------------------------
    print("\n[2] INIT SAE")

    # Получаем параметры модели
    model_cfg = cfg.get("model", {})
    if not model_cfg:
        model_cfg = cfg.get("sae", {})  # для обратной совместимости
    
    input_dim = model_cfg.get("input_dim", data["activations"].shape[1])
    latent_dim = model_cfg.get("latent_dim", model_cfg.get("hidden_dim", 4096))

    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")

    sae = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
    ).to(device)

    # -----------------------------------------------------
    # 5. CHECK CHECKPOINT
    # -----------------------------------------------------
    print("\n[3] CHECK CHECKPOINT")

    ckpt_path = resolve_checkpoint(cfg)
    do_training = True

    if ckpt_path and not args.force_train:
        print(f"  Found: {ckpt_path}")
        
        sae.load_state_dict(torch.load(ckpt_path, map_location=device))

        if args.resume:
            print("  Mode: Resume training")
            do_training = True
        else:
            print("  Mode: Use pretrained (skip training)")
            do_training = False
    elif ckpt_path is None:
        print("  No checkpoint found")
        print("  Mode: Train from scratch")
        do_training = True
    elif args.force_train:
        print("  Mode: Force training from scratch")
        do_training = True

    # -----------------------------------------------------
    # 6. INIT TRAINER
    # -----------------------------------------------------
    print("\n[4] INIT TRAINER")

    # Получаем параметры обучения
    train_cfg = cfg.get("training", {})
    
    lr = train_cfg.get("lr", 5e-4)
    l1_coef = train_cfg.get("l1_coef", 1e-3)
    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 2)
    epochs = train_cfg.get("epochs", 50)

    print(f"  Learning rate: {lr}")
    print(f"  L1 coefficient: {l1_coef}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Epochs: {epochs}")

    # Создаем trainer
    trainer = SAETrainer(
        sae=sae,
        lr=lr,
        l1_coef=l1_coef,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_wandb=False,
    )

    # -----------------------------------------------------
    # 7. TRAIN SAE (OPTIONAL)
    # -----------------------------------------------------
    print("\n[5] TRAINING")

    local_save_dir, google_save_dir = resolve_save_dirs(cfg)

    if do_training:
        print(f"  Save dir: {local_save_dir}")
        print(f"  Google backup: {google_save_dir}")
        
        trainer.fit(
            data=data,
            epochs=epochs,
            save_dir=local_save_dir,
        )

        print("\n[✓] TRAINING COMPLETED")
    else:
        print("  Skipping training (using existing checkpoint)")

    # -----------------------------------------------------
    # 8. COPY TO GOOGLE DRIVE
    # -----------------------------------------------------
    print("\n[6] BACKUP TO GOOGLE DRIVE")

    if google_save_dir and local_save_dir and os.path.exists(local_save_dir):
        os.makedirs(google_save_dir, exist_ok=True)
        
        files_copied = 0
        for fname in os.listdir(local_save_dir):
            src = os.path.join(local_save_dir, fname)
            dst = os.path.join(google_save_dir, fname)
            
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                files_copied += 1
                print(f"  Copied: {fname}")
        
        print(f"  Total files copied: {files_copied}")
    else:
        print("  No backup directory specified")

    print("\n" + "="*50)
    print("DONE!")
    print(f"Models saved in: {local_save_dir}")
    if google_save_dir:
        print(f"Backup in: {google_save_dir}")
    print("="*50)

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()