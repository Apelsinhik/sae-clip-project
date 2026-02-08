# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/utils/backup.py
# -*- coding: utf-8 -*-

import os
import shutil
from datetime import datetime
from multiprocessing import Process

def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _copy_file(src, dst):
    shutil.copy2(src, dst)


def _copy_folder(src, dst):
    shutil.copytree(src, dst)


def backup_file_safe(src, dst_dir, timeout=30):
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src):
        print("[backup skip] src missing")
        return

    name = os.path.basename(src)
    dst = os.path.join(dst_dir, f"{_ts()}_{name}")

    p = Process(target=_copy_file, args=(src, dst))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        print("[backup timeout] file backup killed")
    else:
        print(f"[backup OK] {dst}")


def backup_folder_safe(src, dst_root, timeout=120):
    os.makedirs(dst_root, exist_ok=True)

    if not os.path.exists(src):
        print("[backup skip] src missing")
        return

    name = os.path.basename(src.rstrip("/"))
    dst = os.path.join(dst_root, f"{name}_{_ts()}")

    p = Process(target=_copy_folder, args=(src, dst))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        print("[backup timeout] folder backup killed")
    else:
        print(f"[backup OK] {dst}")
