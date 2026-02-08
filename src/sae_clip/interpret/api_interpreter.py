# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/sae_clip/interpret/api_interpreter.py

import csv
import os
from PIL import Image


class APIInterpreter:
    """
    Интерфейс для автоинтерпретации фичей через VLM/LLM API.

    Идея:
    -----
    Для каждой фичи:
      1) берем top-K картинок
      2) отправляем в модель
      3) получаем 1 фразу-описание
      4) сохраняем в CSV

    Почему интерфейс:
    -----------------
    Чтобы можно было легко поменять backend:
        - OpenRouter
        - OpenAI
        - HuggingFace
        - Локальные модели
    """

    def __init__(self, backend):
        """
        backend — объект с методом .interpret(images) -> str
        где images — список PIL изображений
        """
        self.backend = backend

    def run(self, topk_paths, out_csv):
        """
        topk_paths — dict { feature_id: [paths...] }
        out_csv — куда сохранить CSV

        CSV формат:
        feature_id, interpretation, img_paths
        """

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["feature_id", "interpretation", "sample_paths"])

            for feat, paths in topk_paths.items():
                images = [Image.open(p).convert("RGB") for p in paths]

                # запрос к backend
                interpretation = self.backend.interpret(images)

                # запись
                writer.writerow([
                    feat,
                    interpretation,
                    "|".join(paths)
                ])
