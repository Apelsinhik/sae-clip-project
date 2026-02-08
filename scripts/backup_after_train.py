# FILE: clip-sae-interpret/clip-sae-interpret_clean/scripts/backup_after_train.py
# -*- coding: utf-8 -*-

from sae_clip.utils.ultimate_backup_manager import UltimateBackupManager

backup = UltimateBackupManager(
    google_root="/content/drive/MyDrive",
    yandex_root="/content/yadisk/SAE_PROJECT"
)

backup.backup_folder("./models", "models")
backup.backup_folder("./results", "results")

print("=== BACKUP DONE ===")


