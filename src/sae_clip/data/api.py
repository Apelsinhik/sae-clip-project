# FILE: src/sae_clip/data/api.py
# -*- coding: utf-8 -*-

"""
DATA API

УЛЬТРА СТАБИЛЬНАЯ логика хранения:

READ:
Yandex → Google → Extract

WRITE:
Google only
"""

import os
import torch
from typing import Dict, Any, Tuple

from transformers import CLIPImageProcessor

from sae_clip.models.clip_wrapper import CLIPWrapper
from sae_clip.data.datasets import CIFAR10CLIPDataset
from sae_clip.data.activations import ActivationExtractor

from sae_clip.utils.storage_utils import pick_storage


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def load_or_extract_activations(
    cfg: Dict[str, Any],
    return_path: bool = False
) -> Tuple[Dict[str, torch.Tensor], str]:

    dataset_name = cfg["data"]["dataset"]

    local_activ = cfg["data"]["activations"]

    yandex_root = cfg.get("yandex_root", "")
    google_root = cfg.get("google_root", "")

    yandex_activ = f"{yandex_root}/activations/activations_cifar10_vit_l14.pt"
    google_activ = f"{google_root}/SAE_PROJECT/activations/activations_cifar10_vit_l14.pt"

    # -------------------------------------------------
    # TRY STORAGE READ
    # -------------------------------------------------

    src_path = pick_storage(yandex_activ, google_activ)

    if src_path is not None:
        print(f"[DATA] Loading activations → {src_path}")
        x = torch.load(src_path, map_location="cpu")

        if return_path:
            return {"activations": x}, src_path

        return {"activations": x}

    # -------------------------------------------------
    # EXTRACT NEW
    # -------------------------------------------------

    print("[DATA] Activations not found → extracting CLIP activations")

    device = cfg["data"].get("device", "cuda")

    clip_model = CLIPWrapper(device=device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if dataset_name.lower() == "cifar10":
        dataset = CIFAR10CLIPDataset(
            root=cfg["data"]["root"],
            processor=processor,
            split="train"
        )
    else:
        raise ValueError("Dataset not supported yet")

    extractor = ActivationExtractor(clip_model=clip_model, device=device)

    os.makedirs(os.path.dirname(google_activ), exist_ok=True)

    extractor.extract(
        dataset=dataset,
        save_path=google_activ,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    print(f"[DATA] Saved new activations → Google: {google_activ}")

    x = torch.load(google_activ, map_location="cpu")

    if return_path:
        return {"activations": x}, google_activ

    return {"activations": x}
