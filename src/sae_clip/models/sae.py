# FILE: clip-sae-interpret_clean/src/sae_clip/models/sae.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Простой разреженный автоэнкодер для работы с CLIP-эмбеддингами.
    """

    def __init__(self, input_dim: int = 768, latent_dim: int = 4096):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Линейные слои кодировщика и декодировщика
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

        # Инициализация
        self.reset_parameters()

    def reset_parameters(self):
        """Инициализация весов."""
        nn.init.kaiming_uniform_(self.encoder.weight, a=0.01)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """
        Прямой проход:
        1) projected = encoder(x)
        2) z = ReLU(projected)
        3) x_hat = decoder(z)
        """
        projected = self.encoder(x)
        z = F.relu(projected)  # sparsity через положительные активации
        x_hat = self.decoder(z)
        return x_hat, z

    @staticmethod
    def l1_sparsity(z):
        """L1 sparsity термин: ||z||_1"""
        return torch.mean(torch.abs(z))

    @staticmethod
    def l0_sparsity(z):
        """Оценка L0 sparsity: среднее кол-во ненулевых элементов"""
        return torch.mean((z > 0).float())

    @staticmethod
    def reconstruction_loss(x, x_hat):
        """MSE reconstruction loss: ||x - x_hat||^2"""
        return F.mse_loss(x_hat, x, reduction='mean')