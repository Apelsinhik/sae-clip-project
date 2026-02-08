# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/utils/safe_cloud_backup.py
# -*- coding: utf-8 -*-

import os
import shutil
from datetime import datetime


class SafeCloudBackup:

    def __init__(self, yandex_root=None, google_root=None):
        self.yandex_root = yandex_root
        self.google_root = google_root

    def _ts(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _is_mount_alive(self, path):
        if not path:
            return False
        if not os.path.exists(path):
            return False
        try:
            os.listdir(path)
            return True
        except:
            return False

    def copy_archive(self, archive_path):

        print("\n=== CLOUD BACKUP ===")

        if not os.path.exists(archive_path):
            print("Archive missing:", archive_path)
            return

        # ---------- YANDEX ----------
        if self._is_mount_alive(self.yandex_root):
            try:
                dst = f"{self.yandex_root}/backups"
                os.makedirs(dst, exist_ok=True)

                shutil.copy2(
                    archive_path,
                    f"{dst}/{os.path.basename(archive_path)}"
                )

                print("[OK] Yandex backup")

            except Exception as e:
                print("[FAIL] Yandex:", e)

        else:
            print("[SKIP] Yandex mount dead")

        # ---------- GOOGLE ----------
        if self._is_mount_alive(self.google_root):
            try:
                dst = f"{self.google_root}/SAE_BACKUP"
                os.makedirs(dst, exist_ok=True)

                shutil.copy2(
                    archive_path,
                    f"{dst}/{os.path.basename(archive_path)}"
                )

                print("[OK] Google backup")

            except Exception as e:
                print("[FAIL] Google:", e)

        else:
            print("[SKIP] Google mount dead")
