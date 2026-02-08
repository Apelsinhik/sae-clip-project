# FILE: clip-sae-interpret_clean/src/sae_clip/models/clip_wrapper.py
# -*- coding: utf-8 -*-

import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPWrapper:
    """
    Обёртка над CLIP (ViT-B/32) через библиотеку HuggingFace Transformers.
    
    ВАЖНО: Использует CLIP-base-patch32 (512D эмбеддинги)
    для совместимости с SAE 512D.
    Все вычисления на GPU если доступно.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        # Определяем устройство
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем процессор и модель CLIP (BASE версия - 512D)
        print(f"[CLIPWrapper] Загрузка модели: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

        # По умолчанию CLIP работает в inference mode
        self.model.eval()
        
        # Размерность эмбеддингов (должно быть 512 для CLIP-base)
        self.embedding_dim = self.model.config.projection_dim
        print(f"[CLIPWrapper] Размерность эмбеддингов: {self.embedding_dim}D")
        
        if self.embedding_dim != 512:
            print(f"⚠️  ВНИМАНИЕ: Размерность {self.embedding_dim}D, ожидается 512D для SAE")

    def to(self, device: str):
        """
        Позволяет использовать clip.to(device), как у torch.nn.Module
        """
        self.device = device
        self.model = self.model.to(device)
        return self

    def eval(self):
        """
        Совместимость с torch API: model.eval()
        """
        self.model.eval()
        return self

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor):
        """
        Вычисляет CLS эмбеддинги для батча тензоров (B, 3, 224, 224)
        Возвращает нормализованные 512D эмбеддинги
        """
        # Используем vision_model напрямую для гарантии получения тензора
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        
        # Берем CLS токен (первый токен)
        image_features = vision_outputs.last_hidden_state[:, 0, :]
        
        # Проецируем через visual_projection
        image_features = self.model.visual_projection(image_features)
        
        # Нормализуем
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features

    @torch.no_grad()
    def encode_images(self, images):
        """
        Принимает PIL (или список PIL) → возвращает нормализованные (B, 512)
        Удобно для zero-shot классификации.
        """
        # Обработка через CLIP процессор
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Используем vision_model напрямую для гарантии получения тензора
        vision_outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
        
        # Берем CLS токен (первый токен)
        image_features = vision_outputs.last_hidden_state[:, 0, :]
        
        # Проецируем через visual_projection
        image_features = self.model.visual_projection(image_features)
        
        # Нормализуем
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features

    @torch.no_grad()
    def encode_text(self, text):
        """
        Аналог encode_images, но для текста
        Возвращает нормализованные 512D эмбеддинги
        """
        if not isinstance(text, torch.Tensor):
            # Токенизация текста
            text = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        
        # Используем text_model напрямую для гарантии получения тензора
        text_outputs = self.model.text_model(
            input_ids=text['input_ids'],
            attention_mask=text['attention_mask']
        )
        
        # Берем CLS токен (pooler_output или первый токен)
        if hasattr(text_outputs, 'pooler_output'):
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Проецируем через text_projection
        text_features = self.model.text_projection(text_features)
        
        # Нормализуем
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features