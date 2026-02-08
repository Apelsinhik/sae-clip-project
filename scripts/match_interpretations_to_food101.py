# FILE: clip-sae-interpret_clean/scripts/match_interpretations_to_food101.py
# -*- coding: utf-8 -*-

"""
match_interpretations_to_food101.py

Сопоставление интерпретаций SAE-фич с классами Food-101.

Назначение:
- взять очищенные интерпретации
- сопоставить их с 101 классом Food-101
- получить наиболее вероятный класс для каждой фичи

Используется:
- TF-IDF + cosine similarity
"""

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        classes = [line.strip().lower() for line in f.readlines()]
    return classes


def match_interpretations(df, classes):

    texts = df["clean_text"].fillna("").str.lower().tolist()

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(texts + classes)

    interp_vecs = tfidf_matrix[: len(texts)]
    class_vecs = tfidf_matrix[len(texts) :]

    sims = cosine_similarity(interp_vecs, class_vecs)

    best_classes = []
    best_scores = []

    for row in sims:
        idx = row.argmax()
        best_classes.append(classes[idx])
        best_scores.append(float(row[idx]))

    df["matched_class"] = best_classes
    df["similarity_score"] = best_scores

    return df


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Match SAE interpretations to Food-101 classes"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Путь к CSV с очищенными интерпретациями",
    )

    parser.add_argument(
        "--classes_txt",
        type=str,
        required=True,
        help="Путь к файлу classes.txt из Food-101",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="./results/interpretations_with_classes.csv",
        help="Куда сохранить результат",
    )

    args = parser.parse_args()

    print("[1] Загрузка данных")

    df = pd.read_csv(args.input_csv)

    classes = load_classes(args.classes_txt)

    print(f"Загружено интерпретаций: {len(df)}")
    print(f"Число классов Food-101: {len(classes)}")

    print("[2] Сопоставление интерпретаций с классами")

    df = match_interpretations(df, classes)

    print("[3] Сохранение результата")

    df.to_csv(args.output_csv, index=False)

    print("\n[SAVED]")
    print(f"Результат → {args.output_csv}")
    print("Сопоставление завершено!")


if __name__ == "__main__":
    main()
