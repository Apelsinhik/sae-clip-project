# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/sae_clip/generation/kandinsky.py
# -*- coding: utf-8 -*-

import torch


class KandinskySteerer:
    """
    Класс для управления генерацией Kandinsky2.2 с помощью фич SAE.

    Идея:
    -----
    1) Берем text embedding
    2) Берем feature vector f (из SAE latent)
    3) Делаем:
         emb_plus  = emb + alpha * f
         emb_minus = emb - alpha * f
    4) Передаем в генератор модели
    5) Получаем 3 изображения:
         - оригинал
         - с +фичей
         - с -фичей

    Примечание:
    -----------
    Сам движок Kandinsky мы НЕ реализуем здесь, а подключаем в scripts,
    потому что он может быть:
      - API
      - diffusers
      - локальный
    """

    def __init__(self, clip_model, sae, device=None):
        self.clip_model = clip_model
        self.sae = sae
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.sae.eval()
        self.sae.to(self.device)

    @torch.no_grad()
    def modify_embedding(self, text, feature_id, alpha=1.0):
        """
        Возвращает:
            emb_original
            emb_plus
            emb_minus
        """

        # text embedding
        text_emb = self.clip_model.encode_text([text]).to(self.device)  # (1, D)

        # фича SAE
        basis = torch.zeros((1, self.sae.latent_dim), device=self.device)
        basis[0, feature_id] = 1.0 * alpha

        # декодируем из SAE в CLIP space
        feature_vector = self.sae.decoder(basis)  # (1, D)

        # нормализация (как у CLIP)
        feature_vector = feature_vector / feature_vector.norm(dim=-1, keepdim=True)

        emb_original = text_emb
        emb_plus = text_emb + feature_vector
        emb_minus = text_emb - feature_vector

        return emb_original, emb_plus, emb_minus
