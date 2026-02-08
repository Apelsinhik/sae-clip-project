# FILE: clip-sae-interpret/clip-sae-interpret_clean/scripts/generate_examples.py
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
from PIL import Image

from sae_clip.models.clip_wrapper import CLIPWrapper
from sae_clip.models.sae import SparseAutoencoder
from sae_clip.generation.kandinsky import KandinskySteerer


def dummy_kandinsky_generate(text_embedding):
    """
    Заглушка вместо реальной генерации.
    Твоя реальная генерация будет через Kandinsky diffusers/API.
    """
    img = Image.new("RGB", (256, 256), color="gray")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/interpret_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 1) Загружаем CLIP =====
    print("[1] Загружаем CLIP...")
    clip_model = CLIPWrapper(device=device)

    # ===== 2) Загружаем SAE =====
    print("[2] Загружаем SAE...")
    sae = SparseAutoencoder(
        input_dim=cfg["sae"]["input_dim"],
        latent_dim=cfg["sae"]["latent_dim"]
    )
    sae.load_state_dict(torch.load(cfg["sae"]["weights"], map_location=device))
    sae.to(device)
    sae.eval()

    # ===== 3) Инициализируем steerer =====
    print("[3] Инициализируем steerer...")
    steerer = KandinskySteerer(clip_model, sae, device=device)

    feature_ids = cfg["steering"]["features"]
    prompts = cfg["steering"]["prompts"]
    alpha = cfg["steering"]["alpha"]

    out_dir = cfg["output"]["steering"]
    os.makedirs(out_dir, exist_ok=True)

    for feat in feature_ids:
        for prompt in prompts:
            print(f"\n[steering] prompt='{prompt}' feat={feat}")

            emb_orig, emb_plus, emb_minus = steerer.modify_embedding(
                text=prompt,
                feature_id=feat,
                alpha=alpha
            )

            # ===== 4) Генерация картинок =====

            img_orig = dummy_kandinsky_generate(emb_orig)
            img_plus = dummy_kandinsky_generate(emb_plus)
            img_minus = dummy_kandinsky_generate(emb_minus)

            feat_dir = os.path.join(out_dir, f"feat_{feat}")
            os.makedirs(feat_dir, exist_ok=True)

            img_orig.save(os.path.join(feat_dir, f"{prompt}_orig.png"))
            img_plus.save(os.path.join(feat_dir, f"{prompt}_plus.png"))
            img_minus.save(os.path.join(feat_dir, f"{prompt}_minus.png"))

    print("[done] Steering примеры сохранены.")


if __name__ == "__main__":
    main()
