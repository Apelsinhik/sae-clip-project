# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/utils/ultimate_backup_manager.py

# -*- coding: utf-8 -*-

import os
import shutil
from datetime import datetime


class UltimateBackupManager:

    def __init__(self, google_root=None, yandex_root=None):
        self.google_root = google_root
        self.yandex_root = yandex_root

    def _ts(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _alive(self, path):
        try:
            if path and os.path.exists(path):
                os.listdir(path)
                return True
        except:
            pass
        return False

    def backup_folder(self, src, name):

        ts = self._ts()

        # ---------- GOOGLE ----------
        if self._alive(self.google_root):
            dst = f"{self.google_root}/SAE_BACKUP/{name}_{ts}"
            os.makedirs(dst, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[OK] Google → {dst}")
        else:
            print("[SKIP] Google offline")

        # ---------- YANDEX ----------
        if self._alive(self.yandex_root):
            dst = f"{self.yandex_root}/backups/{name}_{ts}"
            os.makedirs(dst, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[OK] Yandex → {dst}")
        else:
            print("[SKIP] Yandex offline")
