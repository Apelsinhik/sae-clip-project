# FILE: clip-sae-interpret/clip-sae-interpret_clean/scripts/interpret_features.py
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from tqdm.auto import tqdm

from sae_clip.models.clip_wrapper import CLIPWrapper
from sae_clip.models.sae import SparseAutoencoder
from sae_clip.interpret.analyzer import FeatureAnalyzer
from sae_clip.interpret.api_interpreter import APIInterpreter

from sae_clip.data.datasets import CIFAR10Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/interpret_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda"

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

    # ===== 3) Грузим датасет =====
    print("[3] Загружаем датасет...")
    dataset = CIFAR10Dataset(root=cfg["data"]["root"], train=False)

    # ===== 4) Сбор активаций SAE =====
    print("[4] Собираем активации SAE...")
    analyzer = FeatureAnalyzer(sae, clip_model, device=device)
    z_all, images = analyzer.collect_activations(
        dataset,
        batch_size=cfg["data"]["batch_size"]
    )

    # ===== 5) Находим top-k =====
    print("[5] Ищем top-k картинок на фичу...")
    topk = analyzer.topk(z_all, k=cfg["analysis"]["topk"])

    # ===== 6) Сохраняем картинки =====
    print("[6] Сохраняем картинки на диск...")
    out_imgs = cfg["output"]["images"]
    analyzer.save_topk_images(topk, images, out_imgs)

    # ===== 7) Интерпретируем через API =====
    print("[7] Интерпретируем фичи через API...")

    class DummyBackend:
        """пример backend, который просто возвращает заглушку"""
        def interpret(self, images):
            return "placeholder feature description"

    backend = DummyBackend()
    interpreter = APIInterpreter(backend)

    out_csv = cfg["output"]["csv"]
    interpreter.run(topk, out_csv)

    print(f"[done] Интерпретация завершена. CSV: {out_csv}")


if __name__ == "__main__":
    main()
