# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/utils/ultimate_backup.py
# -*- coding: utf-8 -*-

import os
import tarfile
import shutil
import time
from datetime import datetime


class UltimateBackup:
    """
    ULTIMATE BACKUP MANAGER

    Делает:
    1) Архивирует результаты в 1 tar.gz
    2) Копирует архив на Yandex
    3) Делает fallback на Google
    """

    def __init__(
        self,
        yandex_root=None,
        google_root=None,
        tmp_dir="./backup_tmp",
        retries=3
    ):
        self.yandex_root = yandex_root
        self.google_root = google_root
        self.tmp_dir = tmp_dir
        self.retries = retries

        os.makedirs(tmp_dir, exist_ok=True)

        print("\n=== ULTIMATE BACKUP INIT ===")
        print("Yandex:", self._check(yandex_root))
        print("Google:", self._check(google_root))

    def _check(self, path):
        if path and os.path.exists(path):
            return path
        return "OFF"

    def _ts(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------
    # CREATE ARCHIVE
    # ---------------------
    def make_archive(self, sources: dict):

        archive_path = os.path.join(
            self.tmp_dir,
            f"sae_backup_{self._ts()}.tar.gz"
        )

        print("\n[Backup] Creating archive:", archive_path)

        with tarfile.open(archive_path, "w:gz") as tar:
            for name, path in sources.items():
                if path and os.path.exists(path):
                    print(f"  + add {name}")
                    tar.add(path, arcname=name)
                else:
                    print(f"  - skip {name}")

        return archive_path

    # ---------------------
    # SAFE COPY
    # ---------------------
    def _safe_copy(self, src, dst_dir):

        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))

        for attempt in range(self.retries):
            try:
                shutil.copy2(src, dst)
                print("[Backup OK]", dst)
                return True
            except Exception as e:
                print(f"[Retry {attempt+1}] {e}")
                time.sleep(2)

        print("[Backup FAILED]", dst_dir)
        return False

    # ---------------------
    # FULL BACKUP
    # ---------------------
    def backup(self, archive_path):

        success = False

        if self.yandex_root and os.path.exists(self.yandex_root):
            print("\n[Backup] → Yandex")
            success |= self._safe_copy(
                archive_path,
                f"{self.yandex_root}/backups"
            )

        if self.google_root and os.path.exists(self.google_root):
            print("\n[Backup] → Google")
            success |= self._safe_copy(
                archive_path,
                f"{self.google_root}/SAE_BACKUP"
            )

        if not success:
            print("\n[Backup WARNING] No remote backup succeeded")

        return success
