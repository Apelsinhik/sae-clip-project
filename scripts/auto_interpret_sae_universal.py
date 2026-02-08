# FILE: clip-sae-interpret_clean/scripts/auto_interpret_sae_universal.py
# -*- coding: utf-8 -*-

"""
auto_interpret_sae_universal.py

универсальный скрипт интерпретации SAE-фич.

Особенности:

- выбор модели OpenRouter через аргумент
- безопасная обработка ЛЮБЫХ ошибок OpenRouter
- fallback на BLIP-2 только по явному согласию пользователя
- дозапись CSV без перезаписи
- поддержка разных prompt_version
"""

import os
import argparse
import base64
import time
import requests
import pandas as pd
from tqdm import tqdm
from PIL import Image

import sys
import select

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def ask_user_timeout(prompt, timeout=120):
    print("\n" + prompt)
    print(f"(ожидание ответа {timeout} секунд: y/n)")

    try:
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)

        if rlist:
            answer = sys.stdin.readline().strip().lower()
            return answer == "y"
        else:
            print("\n[INFO] Время ожидания ответа истекло.")
            return False

    except Exception:
        print("\n[INFO] Не удалось получить ввод пользователя.")
        return False


def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_existing_csv(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["feature_id", "caption", "interpretation", "source", "prompt_version"])

    try:
        df = pd.read_csv(csv_path)

        if "feature_id" not in df.columns:
            print("[WARN] CSV существует, но не содержит feature_id → создаём новый")
            return pd.DataFrame(columns=["feature_id", "caption", "interpretation", "source", "prompt_version"])

        return df

    except Exception as e:
        print(f"[WARN] Не удалось прочитать CSV: {e}")
        return pd.DataFrame(columns=["feature_id", "caption", "interpretation", "source", "prompt_version"])


# =========================================================
# OPENROUTER
# =========================================================

def query_openrouter(image_b64, api_key, user_prompt, model_name, retries=3):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 40
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"[WARN] Попытка OpenRouter {attempt+1} не удалась: {e}")
            time.sleep(2)

    raise RuntimeError("OPENROUTER_FAILED")


# =========================================================
# BLIP-2
# =========================================================

class LocalBLIP2:
    def __init__(self):
        print("[BLIP-2] Загрузка локальной модели...")

        torch.set_num_threads(1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.model.eval()

        print("[BLIP-2] Модель успешно загружена.")

    @torch.no_grad()
    def caption(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
            do_sample=False
        )

        return self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True
        ).strip()


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Авто-интерпретация SAE-фич"
    )

    parser.add_argument("--images_dir", type=str, required=True)

    parser.add_argument("--num_features", type=int, default=300)

    parser.add_argument("--save_csv", type=str,
                        default="./results/sae_auto_interpretations.csv")

    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Модель OpenRouter для использования"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe in one concise English sentence what visual concept is common across all images in this collage."
    )

    parser.add_argument(
        "--prompt_version",
        type=str,
        default="default"
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)

    df_existing = load_existing_csv(args.save_csv)
    done = set(df_existing["feature_id"].tolist())

    print(f"[INFO] Уже интерпретировано: {len(done)} фич")

    print("[1] Загрузка списка изображений")

    files = sorted(
        [
            f for f in os.listdir(args.images_dir)
            if f.startswith("feature_") and f.endswith(".png")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    files = [f for f in files if int(f.split("_")[1].split(".")[0]) not in done]
    files = files[: args.num_features]

    print(f"[INFO] Необходимо обработать: {len(files)} фич")

    api_key = os.getenv("OPENROUTER_API_KEY")
    use_openrouter = api_key is not None

    if use_openrouter:
        print(f"[MODE] Используется OpenRouter (модель: {args.model})")
    else:
        print("[MODE] OpenRouter недоступен → используется BLIP-2")

    blip2 = None
    results = []

    print("[2] Интерпретация фич")

    for fname in tqdm(files):

        fid = int(fname.split("_")[1].split(".")[0])
        img_path = os.path.join(args.images_dir, fname)

        caption = None
        interpretation = None
        source = None

        if use_openrouter:
            try:
                b64 = encode_image_to_base64(img_path)

                interpretation = query_openrouter(
                    b64,
                    api_key,
                    args.prompt,
                    args.model
                )

                source = "openrouter"

            except RuntimeError:

                agree = ask_user_timeout(
                    "OpenRouter вернул ошибку. Перейти на BLIP-2?"
                )

                if agree:
                    print("[INFO] Переход на BLIP-2")
                    use_openrouter = False
                else:
                    print("[INFO] Работа остановлена пользователем.")
                    break

        if not use_openrouter:
            if blip2 is None:
                blip2 = LocalBLIP2()

            caption = blip2.caption(img_path)
            interpretation = caption
            source = "blip2"

        results.append({
            "feature_id": fid,
            "caption": caption if caption else "",
            "interpretation": interpretation,
            "source": source,
            "prompt_version": args.prompt_version
        })

        if len(results) % 10 == 0:
            df_new = pd.DataFrame(results)
            df_existing = pd.concat([df_existing, df_new])
            df_existing.to_csv(args.save_csv, index=False)
            results = []

    if results:
        df_new = pd.DataFrame(results)
        df_existing = pd.concat([df_existing, df_new])
        df_existing.to_csv(args.save_csv, index=False)

    print("\n[SAVED]")
    print(f"CSV → {args.save_csv}")
    print("DONE")


if __name__ == "__main__":
    main()
