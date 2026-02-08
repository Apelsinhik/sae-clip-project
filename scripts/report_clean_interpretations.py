# FILE: clip-sae-interpret_clean/scripts/report_clean_interpretations.py
# -*- coding: utf-8 -*-

"""
report_clean_interpretations.py

Скрипт для формирования формального отчёта по уже очищенному CSV
с интерпретациями SAE-фич.

Назначение:
- фиксация результатов анализа в виде отдельного файла
- получение статистики по очищенным данным
- сохранение ключевых метрик для отчёта проекта

Вход:
- results/sae_auto_interpretations_clean.csv

Выход:
- results/clean_interpretations_report.txt
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

def word_frequency(texts, top_k=30):
    """
    Подсчёт частоты слов в списке текстов.

    texts  – список строк
    top_k  – сколько самых частых слов вернуть
    """

    # Объединяем все тексты в одну строку
    all_text = " ".join(texts).lower()

    # Извлекаем только буквенные слова
    words = re.findall(r"[a-z']+", all_text)

    # Список стоп-слов, которые не несут смысловой нагрузки
    stopwords = {
        "a", "of", "the", "and", "in", "with",
        "on", "to", "for", "is", "are", "an",
        "that", "this", "across", "all"
    }

    # Убираем стоп-слова
    filtered = [w for w in words if w not in stopwords]

    # Возвращаем top_k самых частых слов
    return Counter(filtered).most_common(top_k)


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # ARGUMENT PARSING
    # -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Формирование отчёта по очищенным интерпретациям SAE"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/sae_auto_interpretations_clean.csv",
        help="Путь к очищенному CSV файлу",
    )

    parser.add_argument(
        "--output_report",
        type=str,
        default="results/clean_interpretations_report.txt",
        help="Путь к итоговому файлу отчёта",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    df = pd.read_csv(args.input_csv)

    # Открываем файл отчёта для записи
    with open(args.output_report, "w", encoding="utf-8") as f:

        # Вспомогательная функция для записи строки в файл и в консоль
        def write(line=""):
            print(line)
            f.write(line + "\n")

        # -------------------------------------------------
        # REPORT HEADER
        # -------------------------------------------------
        write("=== ОТЧЁТ ПО ОЧИЩЕННЫМ ИНТЕРПРЕТАЦИЯМ SAE ===")
        write(f"Дата генерации: {datetime.now()}")
        write("")

        # -------------------------------------------------
        # BASIC STATISTICS
        # -------------------------------------------------
        write("1) БАЗОВАЯ СТАТИСТИКА")
        write("---------------------")

        total = len(df)
        unique = df["clean_text"].nunique()

        write(f"Всего очищенных интерпретаций: {total}")
        write(f"Уникальных интерпретаций: {unique}")

        # Средняя длина интерпретации в словах
        avg_len = df["clean_text"].apply(lambda x: len(str(x).split())).mean()
        write(f"Средняя длина (в словах): {avg_len:.2f}")
        write("")

        # -------------------------------------------------
        # QUALITY SCORE ANALYSIS
        # -------------------------------------------------
        write("2) АНАЛИЗ ОЦЕНКИ КАЧЕСТВА (quality_score)")
        write("-----------------------------------------")

        if "quality_score" in df.columns:
            desc = df["quality_score"].describe()

            # Выводим основные статистики по quality_score
            for k, v in desc.items():
                write(f"{k}: {v}")

        write("")

        # -------------------------------------------------
        # TOP INTERPRETATIONS
        # -------------------------------------------------
        write("3) ТОП-10 ЛУЧШИХ ИНТЕРПРЕТАЦИЙ")
        write("-------------------------------")

        # Выбираем 10 интерпретаций с наибольшим quality_score
        top = df.sort_values("quality_score", ascending=False).head(10)

        for i, t in enumerate(top["clean_text"], 1):
            write(f"{i}. {t}")

        write("")

        # -------------------------------------------------
        # WORD FREQUENCY ANALYSIS
        # -------------------------------------------------
        write("4) НАИБОЛЕЕ ЧАСТЫЕ СЛОВА")
        write("------------------------")

        freqs = word_frequency(df["clean_text"].tolist(), 30)

        for word, count in freqs:
            write(f"{word}: {count}")

        write("")

        # -------------------------------------------------
        # RANDOM SAMPLE
        # -------------------------------------------------
        write("5) ПРИМЕРЫ СЛУЧАЙНЫХ ОЧИЩЕННЫХ ИНТЕРПРЕТАЦИЙ")
        write("--------------------------------------------")

        # Берём случайные 15 примеров (или меньше, если строк мало)
        sample = df["clean_text"].sample(min(15, len(df))).tolist()

        for s in sample:
            write(f"- {s}")

        write("")
        write("=== КОНЕЦ ОТЧЁТА ===")

    # Сообщаем пользователю, что отчёт успешно сохранён
    print("\n[ОТЧЁТ СОХРАНЁН]")
    print(args.output_report)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
