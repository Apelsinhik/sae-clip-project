# FILE: clip-sae-interpret_clean/scripts/train_sae_google_only.py
# -*- coding: utf-8 -*-

"""
Упрощенный скрипт для обучения SAE.
ВСЁ СОХРАНЯЕТ НА GOOGLE DRIVE!
"""

import os
import argparse
import torch
import shutil

from sae_clip.models.sae import SparseAutoencoder
from sae_clip.training.trainer import SAETrainer

def main():
    parser = argparse.ArgumentParser(description="Train SAE on Google Drive")
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["cifar10", "food101"],
                       help="Dataset to train on")
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--l1_coef", type=float, default=0.001)
    parser.add_argument("--latent_dim", type=int, default=4096)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # =====================================================
    # 1. LOAD ACTIVATIONS
    # =====================================================
    print(f"\n[1] LOADING {args.dataset.upper()} ACTIVATIONS")
    
    if args.dataset == "cifar10":
        activations_path = "/content/drive/MyDrive/SAE_PROJECT/activations/activations_cifar10_vit_b32.pt"
        save_dir = "/content/drive/MyDrive/SAE_PROJECT/models/sae_cifar_512d"
    else:  # food101
        activations_path = "/content/drive/MyDrive/SAE_PROJECT/activations/activations_food101_vit_b32.pt"
        save_dir = "/content/drive/MyDrive/SAE_PROJECT/models/sae_food101_512d"
    
    print(f"  Activations: {activations_path}")
    print(f"  Save to: {save_dir}")
    
    # Загружаем данные
    data = torch.load(activations_path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.float16:
            print(f"  Converting Half → Float: {data.shape}")
            data = data.float()  # float16 → float32
        data = {"activations": data}
    
    print(f"  Shape: {data['activations'].shape}")
    input_dim = data["activations"].shape[1]
    
    # =====================================================
    # 2. INIT SAE
    # =====================================================
    print(f"\n[2] INITIALIZING SAE")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {args.latent_dim}")
    
    sae = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    
    # =====================================================
    # 3. INIT TRAINER
    # =====================================================
    print(f"\n[3] INITIALIZING TRAINER")
    print(f"  Learning rate: {args.lr}")
    print(f"  L1 coefficient: {args.l1_coef}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    
    trainer = SAETrainer(
        sae=sae,
        lr=args.lr,
        l1_coef=args.l1_coef,
        batch_size=args.batch_size,
        num_workers=2,
        device=device,
        use_wandb=False,
    )
    
    # =====================================================
    # 4. CREATE SAVE DIRECTORY
    # =====================================================
    print(f"\n[4] CREATING SAVE DIRECTORY")
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Created: {save_dir}")
    
    # =====================================================
    # 5. TRAIN
    # =====================================================
    print(f"\n[5] TRAINING SAE")
    
    trainer.fit(
        data=data,
        epochs=args.epochs,
        save_dir=save_dir,
    )
    
    # =====================================================
    # 6. VERIFY SAVED MODEL
    # =====================================================
    print(f"\n[6] VERIFYING SAVED MODEL")
    
    # Проверяем последний чекпоинт
    checkpoint_path = os.path.join(save_dir, f"sae_epoch_{args.epochs}.pt")
    if os.path.exists(checkpoint_path):
        # Загружаем и проверяем размерности
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        print(f"  ✓ Model keys: {list(state_dict.keys())}")
        
        # Проверяем размерности
        if "encoder.weight" in state_dict:
            encoder_shape = state_dict["encoder.weight"].shape
            decoder_shape = state_dict["decoder.weight"].shape
            print(f"  ✓ Encoder: {encoder_shape} (expected: [{args.latent_dim}, {input_dim}])")
            print(f"  ✓ Decoder: {decoder_shape} (expected: [{input_dim}, {args.latent_dim}])")
    else:
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETED!")
    print(f"Model saved in: {save_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()