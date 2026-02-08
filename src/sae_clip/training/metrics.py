# FILE: clip-sae-interpret_clean/src/sae_clip/training/metrics.py
# -*- coding: utf-8 -*-

import os
import csv
import torch

# Малый epsilon для численной стабильности
EPS = 1e-8


# =========================================================
# Базовые метрики
# =========================================================

def mse_loss(x, x_hat):
    """
    Mean Squared Error (MSE) между входом и реконструкцией.
    """
    return torch.mean((x - x_hat) ** 2).item()


def l0_sparsity(z, threshold: float = 1e-6):
    """
    L0 sparsity (аппроксимация):
    доля латентов, модуль которых > threshold.
    """
    return torch.mean((torch.abs(z) > threshold).float()).item()


def explained_variance(x, x_hat):
    """
    Explained Variance Ratio (EVR):

        EVR = 1 - Var(x - x_hat) / Var(x)
    """
    residual = x - x_hat
    var_residual = torch.var(residual, dim=0).mean()
    var_original = torch.var(x, dim=0).mean()

    evr = 1.0 - var_residual / (var_original + EPS)
    return evr.item()


def r2_score(x, x_hat):
    """
    R² — коэффициент детерминации:

        R² = 1 - SS_res / SS_tot
    """
    ss_res = torch.sum((x - x_hat) ** 2)
    ss_tot = torch.sum((x - x.mean(dim=0)) ** 2)

    r2 = 1.0 - ss_res / (ss_tot + EPS)
    return r2.item()


# =========================================================
# Унифицированный расчёт метрик
# =========================================================

def compute_all_metrics(
    x,
    x_hat,
    z,
    total_loss: float = None,
    l1_loss: float = None,
):
    """
    Считает все метрики SAE за один вызов.

    Возвращает dict, пригодный для CSV и логирования.
    """

    metrics = {
        "mse": mse_loss(x, x_hat),
        "l0": l0_sparsity(z),
        "evr": explained_variance(x, x_hat),
        "r2": r2_score(x, x_hat),
    }

    if total_loss is not None:
        metrics["loss"] = float(total_loss)

    if l1_loss is not None:
        metrics["l1"] = float(l1_loss)

    return metrics


# =========================================================
# CSV + Console логирование
# =========================================================

def log_metrics_csv(
    metrics: dict,
    epoch: int,
    log_path: str = "logs/sae_metrics.csv"
):
    """
    Записывает метрики в CSV и печатает в консоль.
    """

    # Создаём папку logs/, если её нет
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    row = {"epoch": epoch}
    row.update(metrics)

    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    # Печать в консоль
    log_str = f"[Epoch {epoch}] " + " | ".join(
        [f"{k}: {v:.6f}" for k, v in metrics.items()]
    )
    print(log_str)
