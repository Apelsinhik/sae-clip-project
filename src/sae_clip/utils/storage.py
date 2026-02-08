# FILE: clip-sae-interpret_clean/src/sae_clip/utils/storage.py
# -*- coding: utf-8 -*-

"""
Утилиты для работы с хранилищем (Google Drive / Яндекс Диск)
ВСЁ СОХРАНЯЕМ НА GOOGLE DRIVE!
"""

import os

# =========================================================
# RESOLVE PATHS
# =========================================================

def resolve_activations_path(cfg):
    """
    Находит путь к файлу с активациями.
    Сначала проверяет Google Drive, затем Яндекс Диск.
    """
    storage = cfg["storage"]
    data_cfg = cfg["data"]
    
    file_name = data_cfg["activations_file"]
    
    # 1. Проверяем Google Drive
    google_root = storage.get("google_root")
    if google_root:
        google_path = os.path.join(google_root, "activations", file_name)
        if os.path.exists(google_path):
            print(f"[STORAGE] Using Google → {google_path}")
            return google_path
    
    # 2. Проверяем Яндекс Диск
    yandex_root = storage.get("yandex_root")
    if yandex_root:
        yandex_path = os.path.join(yandex_root, "activations", file_name)
        if os.path.exists(yandex_path):
            print(f"[STORAGE] Using Yandex → {yandex_path}")
            return yandex_path
    
    # 3. Ошибка
    raise FileNotFoundError(
        f"Activation file '{file_name}' not found in Google Drive or Yandex Disk"
    )

def resolve_checkpoint(cfg):
    """
    Находит checkpoint для SAE.
    Проверяет только Google Drive (Яндекс только для чтения).
    """
    storage = cfg["storage"]
    saving_cfg = cfg.get("saving", {})
    
    checkpoints_dir = saving_cfg.get("checkpoints_dir", "models/sae")
    model_name = saving_cfg.get("model_name", "sae")
    
    # ⚡ ВАЖНО: Проверяем только Google Drive
    google_root = storage.get("google_root")
    if google_root:
        # Если checkpoints_dir уже полный путь
        if checkpoints_dir.startswith("/"):
            google_dir = checkpoints_dir
        else:
            google_dir = os.path.join(google_root, checkpoints_dir)
        
        # Ищем последний checkpoint
        if os.path.exists(google_dir):
            # Ищем файлы типа sae_epoch_*.pt
            checkpoint_files = []
            for fname in os.listdir(google_dir):
                if fname.startswith(model_name) and fname.endswith(".pt"):
                    checkpoint_files.append(fname)
            
            if checkpoint_files:
                # Сортируем по номеру эпохи
                checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                latest = checkpoint_files[-1]
                return os.path.join(google_dir, latest)
    
    return None

def resolve_save_dirs(cfg):
    """
    Возвращает папки для сохранения.
    ВСЁ СОХРАНЯЕМ НА GOOGLE DRIVE!
    """
    storage = cfg["storage"]
    saving_cfg = cfg.get("saving", {})
    
    checkpoints_dir = saving_cfg.get("checkpoints_dir", "models/sae")
    
    # ⚡ ВАЖНО: Всегда сохраняем на Google Drive
    google_root = storage["google_root"]
    
    # Если checkpoints_dir уже полный путь
    if checkpoints_dir.startswith("/"):
        google_save_dir = checkpoints_dir
    else:
        google_save_dir = os.path.join(google_root, checkpoints_dir)
    
    # Локальная папка = Google папка
    local_save_dir = google_save_dir
    
    print(f"[STORAGE] Save directory: {local_save_dir}")
    print(f"[STORAGE] Google backup: {google_save_dir}")
    
    return local_save_dir, google_save_dir