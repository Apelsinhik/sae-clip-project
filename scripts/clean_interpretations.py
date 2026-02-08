# FILE: clip-sae-interpret_clean/scripts/clean_interpretations.py
# -*- coding: utf-8 -*-

"""
clean_interpretations.py

Скрипт очистки и нормализации CSV с авто-интерпретациями SAE-фич.

Используется для:
- удаления дубликатов
- удаления шаблонных фраз BLIP-2 / VLM
- нормализации текстов
- оценки качества интерпретаций

Input:
- results/sae_auto_interpretations.csv

Output:
- results/sae_auto_interpretations_clean.csv
- results/cleaning.log
"""

import os
import re
import argparse
import pandas as pd
from datetime import datetime


# =========================================================
# LOGGING
# =========================================================

def log(msg, log_file):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# =========================================================
# TEXT PROCESSING
# =========================================================

def normalize_text(text: str) -> str:
    """
    Базовая нормализация:
    - lower case
    - удаление лишних пробелов
    """
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_templates(text: str) -> str:
    """
    Удаление типичных шаблонных фраз,
    которые мешают смысловой интерпретации.
    """

    patterns = [
        r"a collage of",
        r"collage of",
        r"pictures of",
        r"images of",
        r"images showing",
        r"pictures showing",
        r"various",
        r"different types of",
        r"several",
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def quality_score(text: str) -> int:
    """
    Простая эвристическая оценка качества интерпретации.

    Чем меньше шаблонных слов и чем более
    конкретный текст – тем выше score.
    """

    bad_words = [
        "collage",
        "pictures",
        "images",
        "various",
        "different",
        "showing",
    ]

    score = 10

    for w in bad_words:
        if w in text:
            score -= 2

    length = len(text.split())

    if length < 5:
        score -= 3

    if length > 20:
        score -= 2

    return max(score, 0)


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # ARGPARSE
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Clean and normalize SAE feature interpretations CSV"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/sae_auto_interpretations.csv",
        help="Исходный CSV с интерпретациями",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/sae_auto_interpretations_clean.csv",
        help="Очищенный CSV",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="results/cleaning.log",
        help="Файл логирования",
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    log("\n=== Cleaning SAE Interpretations ===", args.log_file)
    log(f"Run at: {datetime.now()}", args.log_file)

    # -----------------------------------------------------
    # 1. LOAD CSV
    # -----------------------------------------------------
    log(f"[1] Loading CSV: {args.input_csv}", args.log_file)

    df = pd.read_csv(args.input_csv)
    log(f"Total rows: {len(df)}", args.log_file)

    # -----------------------------------------------------
    # 2. NORMALIZATION
    # -----------------------------------------------------
    log("[2] Normalizing texts", args.log_file)

    df["clean_text"] = df["interpretation"].astype(str).apply(normalize_text)

    # -----------------------------------------------------
    # 3. REMOVE TEMPLATES
    # -----------------------------------------------------
    log("[3] Removing template phrases", args.log_file)

    df["clean_text"] = df["clean_text"].apply(remove_templates)

    # -----------------------------------------------------
    # 4. QUALITY SCORE
    # -----------------------------------------------------
    log("[4] Computing quality scores", args.log_file)

    df["quality_score"] = df["clean_text"].apply(quality_score)

    # -----------------------------------------------------
    # 5. DROP EMPTY
    # -----------------------------------------------------
    log("[5] Removing empty / too short entries", args.log_file)

    before = len(df)
    df = df[df["clean_text"].str.len() > 5]

    log(f"Removed: {before - len(df)} rows", args.log_file)

    # -----------------------------------------------------
    # 6. DROP DUPLICATES
    # -----------------------------------------------------
    log("[6] Removing duplicates", args.log_file)

    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"])

    log(f"Duplicates removed: {before - len(df)}", args.log_file)

    # -----------------------------------------------------
    # 7. SORT BY QUALITY
    # -----------------------------------------------------
    log("[7] Sorting by quality score", args.log_file)

    df = df.sort_values(by="quality_score", ascending=False)

    # -----------------------------------------------------
    # 8. SAVE RESULTS
    # -----------------------------------------------------
    log("[8] Saving cleaned CSV", args.log_file)

    df.to_csv(args.output_csv, index=False)

    log(f"Final rows: {len(df)}", args.log_file)
    log(f"Saved to: {args.output_csv}", args.log_file)
    log("Cleaning complete.\n", args.log_file)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
