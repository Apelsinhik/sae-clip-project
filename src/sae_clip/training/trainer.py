# FILE: clip-sae-interpret_clean/src/sae_clip/training/trainer.py
# -*- coding: utf-8 -*-

"""
Trainer для Sparse Autoencoder (SAE)
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SAETrainer:
    """
    Класс для обучения Sparse Autoencoder (SAE)
    """

    def __init__(
        self,
        sae,
        lr=5e-4,
        l1_coef=1e-3,
        batch_size=256,
        num_workers=2,
        device="cuda",
        use_wandb=False,
    ):
        # Проверяем типы параметров
        if not isinstance(lr, (int, float)):
            raise TypeError(f"lr должен быть числом, получен {type(lr)}: {lr}")
        if not isinstance(l1_coef, (int, float)):
            raise TypeError(f"l1_coef должен быть числом, получен {type(l1_coef)}: {l1_coef}")
        
        self.sae = sae
        self.lr = float(lr)
        self.l1_coef = float(l1_coef)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device = device
        self.use_wandb = use_wandb

        # Инициализация оптимизатора
        self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.lr)
        
        # Планировщик learning rate
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,
            eta_min=1e-6
        )

        print(f"[TRAINER] Initialized:")
        print(f"  Learning rate: {self.lr}")
        print(f"  L1 coefficient: {self.l1_coef}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Device: {self.device}")

    def fit(self, data, epochs, save_dir=None):
        """
        Обучает SAE на данных CLIP-активаций.
        """
        # Подготовка данных
        if isinstance(data, torch.Tensor):
            activations = data
        elif isinstance(data, dict) and "activations" in data:
            activations = data["activations"]
        else:
            raise ValueError(f"Неизвестный формат данных: {type(data)}")

        print(f"[TRAINER] Training data: {activations.shape}")
        
        # Проверяем совместимость размерностей
        if activations.shape[1] != self.sae.input_dim:
            print(f"⚠️  Dimension mismatch!")
            print(f"   Data: {activations.shape[1]}D")
            print(f"   SAE: {self.sae.input_dim}D")
            
            # Автоматическая корректировка
            if activations.shape[1] > self.sae.input_dim:
                print(f"   Trimming to {self.sae.input_dim}D")
                activations = activations[:, :self.sae.input_dim]
            else:
                print(f"   Padding to {self.sae.input_dim}D")
                padding = torch.zeros(
                    activations.shape[0], 
                    self.sae.input_dim - activations.shape[1],
                    dtype=activations.dtype
                )
                activations = torch.cat([activations, padding], dim=1)

        # Создаем DataLoader
        dataset = TensorDataset(activations)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda")
        )

        print(f"[TRAINER] Starting training:")
        print(f"  Epochs: {epochs}")
        print(f"  Batches per epoch: {len(loader)}")
        print(f"  Total batches: {epochs * len(loader)}")

        # Цикл обучения
        for epoch in range(1, epochs + 1):
            self.sae.train()
            
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0
            epoch_l0 = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in pbar:
                # Получаем данные
                x = batch[0].to(self.device)

                # Forward pass
                x_hat, z = self.sae(x)

                # Вычисление потерь
                recon_loss = self.sae.reconstruction_loss(x, x_hat)
                l1_loss = self.sae.l1_sparsity(z) * self.l1_coef
                loss = recon_loss + l1_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Сбор статистики
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_l1 += l1_loss.item()
                epoch_l0 += self.sae.l0_sparsity(z).item()

                # Обновление progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "l1": f"{l1_loss.item():.4f}",
                })

            # Средние значения за эпоху
            avg_loss = epoch_loss / len(loader)
            avg_recon = epoch_recon / len(loader)
            avg_l1 = epoch_l1 / len(loader)
            avg_l0 = epoch_l0 / len(loader)

            print(f"[Epoch {epoch:3d}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Recon: {avg_recon:.4f} | "
                  f"L1: {avg_l1:.4f} | "
                  f"L0: {avg_l0:.4f}")

            # Сохранение чекпоинта
            if save_dir and (epoch % 10 == 0 or epoch == epochs):
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"sae_epoch_{epoch}.pt")
                torch.save(self.sae.state_dict(), checkpoint_path)
                print(f"  [SAVE] Checkpoint: {checkpoint_path}")

            # Обновление learning rate
            self.scheduler.step()

        print("[TRAINER] Training completed!")

    def evaluate(self, data_loader):
        """
        Оценка SAE на валидационных данных.
        """
        self.sae.eval()
        
        total_loss = 0.0
        total_recon = 0.0
        total_l1 = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                x = batch[0].to(self.device)
                
                x_hat, z = self.sae(x)
                recon_loss = self.sae.reconstruction_loss(x, x_hat)
                l1_loss = self.sae.l1_sparsity(z) * self.l1_coef
                loss = recon_loss + l1_loss

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_l1 += l1_loss.item()
                total_batches += 1

        return {
            "loss": total_loss / total_batches,
            "recon_loss": total_recon / total_batches,
            "l1_loss": total_l1 / total_batches,
        }