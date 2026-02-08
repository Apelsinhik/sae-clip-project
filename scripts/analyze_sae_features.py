# FILE: clip-sae-interpret_clean/scripts/analyze_sae_features.py
# -*- coding: utf-8 -*-

"""
analyze_sae_features.py

Извлекает top-K изображений для каждой SAE-фичи
по сохранённым CLIP-активациям.

Используется для:
- интерпретации SAE-фич
- авто-интерпретации через VLM (OpenRouter)
- визуализации (коллажи)

Результат:
- CSV файл с:
    feature_id, rank, image_index, activation
"""

import os
import argparse
import torch
import csv
from tqdm import tqdm

from sae_clip.models.sae import SparseAutoencoder


# =========================================================
# MAIN
# =========================================================

def main():
    # -----------------------------------------------------
    # ARGPARSE
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Extract top-K activations for SAE features"
    )

    parser.add_argument(
        "--activations_path",
        type=str,
        required=True,
        help="Путь к сохранённым CLIP-активациям (.pt)",
    )

    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Путь к checkpoint SAE (.pt)",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Сколько изображений брать для каждой фичи",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/sae_features",
        help="Куда сохранять CSV с результатами",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------
    # 1. LOAD ACTIVATIONS
    # -----------------------------------------------------
    print("[1] Loading activations")

    data = torch.load(args.activations_path, map_location="cpu")

    # Активации могут быть:
    # 1) просто Tensor [N, D]
    # 2) dict {"activations": Tensor, ...}
    if isinstance(data, torch.Tensor):
        X = data
    else:
        X = data["activations"]

    print(f"Activations shape: {X.shape}")  # [N, D]

    # -----------------------------------------------------
    # 2. LOAD SAE (КОРРЕКТНО ИЗ CHECKPOINT)
    # -----------------------------------------------------
    print("[2] Loading SAE")

    state_dict = torch.load(args.sae_path, map_location="cpu")

    # ВАЖНО:
    # размеры SAE вытаскиваем из весов энкодера
    encoder_weight = state_dict["encoder.weight"]
    latent_dim, input_dim = encoder_weight.shape

    print(f"SAE dims: input_dim={input_dim}, latent_dim={latent_dim}")

    sae = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
    )

    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()

    # -----------------------------------------------------
    # 3. COMPUTE SAE ACTIVATIONS
    # -----------------------------------------------------
    print("[3] Computing SAE activations")

    with torch.no_grad():
        # Z: [N, F] — активации SAE-фич
        X = X.float()
        _, Z = sae(X.to(device))
        Z = Z.cpu()

    num_features = Z.shape[1]
    print(f"Number of SAE features: {num_features}")

    # -----------------------------------------------------
    # 4. EXTRACT TOP-K АКТИВАЦИЙ
    # -----------------------------------------------------
    print("[4] Extracting top-K per feature")

    results = []

    for feature_id in tqdm(range(num_features)):
        values = Z[:, feature_id]

        # top-K самых сильных активаций
        top_vals, top_idx = torch.topk(values, args.top_k)

        for rank in range(args.top_k):
            results.append({
                "feature_id": feature_id,
                "rank": rank + 1,
                "image_index": int(top_idx[rank]),
                "activation": float(top_vals[rank]),
            })

    # -----------------------------------------------------
    # 5. SAVE CSV
    # -----------------------------------------------------
    csv_path = os.path.join(args.save_dir, "sae_topk_activations.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature_id", "rank", "image_index", "activation"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n[SAVED]")
    print(f"CSV → {csv_path}")
    print("DONE")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
