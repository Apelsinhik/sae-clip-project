# src/sae_clip/storage_wrapper.py

from pathlib import Path
import os
import subprocess

from storage.storage_manager import StorageManager


class StorageWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.manager = StorageManager(cfg)

        # Базовые пути проекта
        self.google_root = Path("/content/drive/MyDrive/SAE_PROJECT")
        self.yandex_root = Path("/content/yadisk/SAE_PROJECT")

    # Проверка существования файла или папки
    def _exists(self, path):
        return os.path.exists(path)

    # ----------- ДАТАСЕТЫ -----------
    # Google → Yandex → скачивание (без копирования)
    def ensure_dataset(self, name):
        local = self.google_root / "datasets" / name
        yandex = self.yandex_root / "datasets" / name

        if self._exists(local):
            return local

        if self._exists(yandex):
            return yandex

        # Если нигде нет – скачиваем в Google Drive
        subprocess.run(["python", "scripts/download_dataset.py", name, str(local)])
        return local

    # ----------- АКТИВАЦИИ -----------
    # Google → Yandex → копирование → вычисление
    def ensure_activations(self, name):
        local = self.google_root / "activations" / f"{name}.pt"
        yandex = self.yandex_root / "activations" / f"{name}.pt"

        if self._exists(local):
            return local

        if self._exists(yandex):
            self._copy_from_yandex(yandex, local)
            return local

        # Если нигде нет – извлекаем активации
        if name == "food101":
            subprocess.run(["python", "scripts/extract_clip_activations_food101.py"])
        else:
            subprocess.run(["python", "scripts/extract_clip_activations_cifar10.py"])

        return local

    # ----------- МОДЕЛИ -----------
    # Google → Yandex → копирование → обучение
    def ensure_model(self, name):
        local = self.google_root / "models" / f"{name}.pt"
        yandex = self.yandex_root / "models" / f"{name}.pt"

        if self._exists(local):
            return local

        if self._exists(yandex):
            self._copy_from_yandex(yandex, local)
            return local

        # Если нигде нет – запускаем обучение
        subprocess.run(["python", "scripts/train_sae.py", name])
        return local

    # ----------- РЕЗУЛЬТАТЫ -----------
    # Все результаты всегда сохраняются только в Google Drive
    def results_path(self, filename):
        path = self.google_root / "results" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # Копирование файла из Яндекс.Диска в Google Drive через StorageManager
    def _copy_from_yandex(self, remote, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        self.manager.pull_file(str(remote), str(local))
