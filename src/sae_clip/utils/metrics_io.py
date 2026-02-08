# FILE: clip-sae-interpret/clip-sae-interpret_clean/src/utils/metrics_io.py
# -*- coding: utf-8 -*-

import os
import json
import csv
from datetime import datetime


def save_zero_shot_results(results: dict, save_dir: str):
    """
    Сохраняет zero-shot метрики в JSON и CSV.

    results: dict
        {
          "cifar10": {
              "clip": float,
              "clip_plus_sae": float | None
          },
          "food101": {
              "clip": float,
              "clip_plus_sae": float | None
          }
        }
    """

    os.makedirs(save_dir, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    # ---------- JSON ----------
    json_path = os.path.join(save_dir, "zero_shot_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # ---------- CSV ----------
    csv_path = os.path.join(save_dir, "zero_shot_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "accuracy"])

        for dataset, vals in results.items():
            for model_name, acc in vals.items():
                if acc is not None:
                    writer.writerow([dataset, model_name, acc])

    print(f"[METRICS] Saved JSON → {json_path}")
    print(f"[METRICS] Saved CSV  → {csv_path}")
