# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/storage/checkpoint_resolver.py
# -*- coding: utf-8 -*-

"""
Checkpoint Resolver

LOGIC:
Yandex → Google → None

READ ONLY from Yandex
READ/WRITE Google
"""

import os
from typing import Optional


def find_last_checkpoint(folder: str) -> Optional[str]:
    """
    Find latest checkpoint by epoch number
    """

    if not os.path.exists(folder):
        return None

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".pt") and "epoch" in f
    ]

    if not files:
        return None

    def epoch_num(name):
        try:
            return int(name.split("epoch_")[-1].split(".")[0])
        except:
            return -1

    files.sort(key=epoch_num)

    return os.path.join(folder, files[-1])


# -------------------------------------------------


def resolve_checkpoint(cfg) -> Optional[str]:

    yandex_root = cfg.get("yandex_root")
    google_root = cfg.get("google_root")

    yandex_ckpt = None
    google_ckpt = None

    # -----------------------
    # YANDEX
    # -----------------------

    if yandex_root:

        yandex_path = f"{yandex_root}/checkpoints"

        ckpt = find_last_checkpoint(yandex_path)

        if ckpt:
            print(f"[CKPT] Yandex → {ckpt}")
            return ckpt

    # -----------------------
    # GOOGLE
    # -----------------------

    if google_root:

        google_path = f"{google_root}/SAE_PROJECT/models/sae"

        ckpt = find_last_checkpoint(google_path)

        if ckpt:
            print(f"[CKPT] Google → {ckpt}")
            return ckpt

    print("[CKPT] None found → train from scratch")

    return None
