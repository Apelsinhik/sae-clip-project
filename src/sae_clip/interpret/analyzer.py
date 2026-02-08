# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/sae_clip/interpret/analyzer.py
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm


class FeatureAnalyzer:
    """
    Анализ активностей фичей в SAE.

    Возможности:
    ------------
    1) собирает матрицу активаций z для всего датасета
    2) находит top-k примеров на каждую фичу
    3) сохраняет результаты для визуализации и интерпретации
    """

    def __init__(self, sae, clip_model, device=None):
        self.sae = sae
        self.clip_model = clip_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sae.to(self.device)
        self.sae.eval()

    @torch.no_grad()
    def collect_activations(self, dataset, batch_size=64):
        """
        Возвращает:
            z_all: (N, L) — матрица латентных активаций
            images: список PIL изображений (для коллажей)
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        z_all = []
        imgs = []

        for images, _ in tqdm(loader, desc="Сбор активаций SAE"):
            embeds = self.clip_model.encode_images(images).to(self.device)
            _, z = self.sae(embeds)
            z_all.append(z.cpu())
            imgs.extend(images)

        z_all = torch.cat(z_all, dim=0)  # (N, L)
        return z_all, imgs

    @staticmethod
    def topk(z_all, k=10):
        """
        Возвращает топ-k индексов для каждой фичи.

        z_all: (N, L)
        result: dict { feature_id: [idx1, idx2, ... idx_k] }
        """
        N, L = z_all.shape
        result = {}

        for feat in range(L):
            values = z_all[:, feat]
            topk_indices = torch.topk(values, k).indices.tolist()
            result[feat] = topk_indices

        return result

    @staticmethod
    def save_topk_images(topk_indices, images, out_dir):
        """
        Сохраняет топ-k картинок для каждой фичи.
        """
        os.makedirs(out_dir, exist_ok=True)

        for feat, idxs in topk_indices.items():
            feat_dir = os.path.join(out_dir, f"feature_{feat}")
            os.makedirs(feat_dir, exist_ok=True)

            for i, idx in enumerate(idxs):
                img = images[idx]
                img.save(os.path.join(feat_dir, f"{i}.png"))
