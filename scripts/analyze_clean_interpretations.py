# FILE: clip-sae-interpret_clean/scripts/analyze_clean_interpretations.py
# -*- coding: utf-8 -*-

"""
analyze_clean_interpretations.py

Скрипт анализа уже очищенного CSV с интерпретациями SAE-фич.

Назначение:
- оценка качества очистки
- анализ статистики по интерпретациям
- подсчёт уникальности
- частотный анализ слов
- вывод аналитического отчёта в лог-файл

Вход:
- results/sae_auto_interpretations_clean.csv

Выход:
- results/clean_analysis.log
"""

import os
import argparse
import pandas as pd
import re
from collections import Counter
from datetime import datetime


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def log(msg, log_file):
    """
    Запись строки одновременно в консоль и в лог-файл
    """
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def word_frequency(texts, top_k=30):
    """
    Подсчёт частоты слов в очищенных интерпретациях
    """

    all_text = " ".join(texts).lower()

    # Извлекаем только слова из латинских букв
    words = re.findall(r"[a-z']+", all_text)

    # Стоп-слова, которые не несут смысловой нагрузки
    stopwords = {
        "a", "of", "the", "and", "in", "with",
        "on", "to", "for", "is", "are", "an",
        "that", "this", "across", "all"
    }

    filtered = [w for w in words if w not in stopwords]

    return Counter(filtered).most_common(top_k)


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # ARGUMENT PARSING
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Анализ очищенных интерпретаций SAE"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/sae_auto_interpretations_clean.csv",
        help="Путь к очищенному CSV файлу",
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="results/clean_analysis.log",
        help="Файл для записи результатов анализа",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    log("\n=== АНАЛИЗ ОЧИЩЕННЫХ ИНТЕРПРЕТАЦИЙ ===", args.log_file)
    log(f"Дата запуска: {datetime.now()}", args.log_file)

    log(f"[1] Загрузка CSV файла: {args.input_csv}", args.log_file)

    df = pd.read_csv(args.input_csv)

    log(f"Всего строк в очищенном файле: {len(df)}", args.log_file)

    # -----------------------------------------------------
    # BASIC STATISTICS
    # -----------------------------------------------------
    log("\n[2] БАЗОВАЯ СТАТИСТИКА", args.log_file)

    unique = df["clean_text"].nunique()
    log(f"Количество уникальных интерпретаций: {unique}", args.log_file)

    avg_len = df["clean_text"].apply(lambda x: len(str(x).split())).mean()
    log(f"Средняя длина интерпретации (в словах): {avg_len:.2f}", args.log_file)

    # -----------------------------------------------------
    # QUALITY SCORE ANALYSIS
    # -----------------------------------------------------
    log("\n[3] АНАЛИЗ ОЦЕНКИ КАЧЕСТВА", args.log_file)

    if "quality_score" in df.columns:
        stats = df["quality_score"].describe()

        log("Статистика по quality_score:", args.log_file)

        for key, value in stats.items():
            log(f"{key}: {value}", args.log_file)

        # Выводим несколько самых качественных интерпретаций
        top_quality = df.sort_values("quality_score", ascending=False).head(10)

        log("\nТоп-10 наиболее качественных интерпретаций:", args.log_file)

        for t in top_quality["clean_text"]:
            log(f" - {t}", args.log_file)

    # -----------------------------------------------------
    # WORD FREQUENCY ANALYSIS
    # -----------------------------------------------------
    log("\n[4] ЧАСТОТНЫЙ АНАЛИЗ СЛОВ", args.log_file)

    freqs = word_frequency(df["clean_text"].tolist(), 30)

    log("Наиболее часто встречающиеся слова:", args.log_file)

    for word, count in freqs:
        log(f"{word}: {count}", args.log_file)

    # -----------------------------------------------------
    # RANDOM SAMPLE
    # -----------------------------------------------------
    log("\n[5] ПРИМЕРЫ ОЧИЩЕННЫХ ИНТЕРПРЕТАЦИЙ", args.log_file)

    sample = df["clean_text"].sample(min(10, len(df))).tolist()

    for s in sample:
        log(f" - {s}", args.log_file)

    log("\nАнализ успешно завершён.\n", args.log_file)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
