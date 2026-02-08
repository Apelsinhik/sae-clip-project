# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/sae_clip/data/activations.py
# -*- coding: utf-8 -*-
"""
activations.py — сбор CLIP активаций
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


class ActivationExtractor:
    def __init__(self, clip_model, device="cuda"):
        self.clip_model = clip_model.to(device)
        self.device = device

    def extract(self, dataset, save_path, batch_size=64, num_workers=2, pin_memory=True):
        """
        dataset: CLIP-ready dataset, возвращает тензор pixel_values
        save_path: куда сохранить CLS эмбеддинги
        """

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        activations = []

        print("[2] Снимаем CLIP активации...")

        self.clip_model.eval()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Снимаем активиации CLIP"):
                batch = batch.to(self.device)

                # CLIP forward
                feats = self.clip_model.get_image_features(batch)

                # Нормализуем и сохраняем CLS в cpu
                feats = feats / feats.norm(dim=-1, keepdim=True)
                activations.append(feats.cpu())

        activations = torch.cat(activations, dim=0)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(activations, save_path)

        print(f"[✓] Активации сохранены: {save_path} → shape={activations.shape}")
