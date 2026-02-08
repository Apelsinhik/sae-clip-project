# FILE: sae_clip/data/datasets.py
# -*- coding: utf-8 -*-

"""
datasets.py — датасеты для CLIP и SAE.

Типы датасетов:
1) *CLIPDataset — для обучения SAE (возвращают pixel_values)
2) *ZeroShotDataset — для zero-shot evaluation (возвращают image, label)
"""

import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Food101, CIFAR10


# =========================================================
# 1. ДАТАСЕТЫ ДЛЯ ОБУЧЕНИЯ SAE
# =========================================================

class CIFAR10CLIPDataset(Dataset):
    """
    CIFAR-10 → CLIPImageProcessor → pixel_values
    Используется ТОЛЬКО для обучения SAE.
    """

    def __init__(self, root, processor, split="train"):
        self.processor = processor
        self.ds = CIFAR10(
            root=root,
            train=(split == "train"),
            download=False
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        pixel_values = self.processor(
            images=img,
            return_tensors="pt"
        )["pixel_values"][0]
        return pixel_values


class Food101CLIPDataset(Dataset):
    """
    Food-101 → CLIPImageProcessor → pixel_values
    Используется ТОЛЬКО для обучения SAE.
    """

    def __init__(self, root, processor, split="train"):
        self.processor = processor
        self.ds = Food101(
            root=root,
            split=split,
            download=False
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        pixel_values = self.processor(
            images=img,
            return_tensors="pt"
        )["pixel_values"][0]
        return pixel_values


# =========================================================
# 2. ZERO-SHOT DATASETS
# =========================================================

class CIFAR10ZeroShotDataset(Dataset):
    """
    CIFAR-10 zero-shot loader (RAW cifar-10-batches-py).

    Ожидаемая структура:
        root/
          data_batch_1
          ...
          test_batch
    """

    def __init__(self, root, train=False, transform=None):
        self.root = root
        self.transform = transform

        self.data = []
        self.labels = []

        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]

        for fname in files:
            path = os.path.join(root, fname)
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")
                self.data.append(entry[b"data"])
                self.labels.extend(entry[b"labels"])

        self.data = np.concatenate(self.data, axis=0)
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # CHW → HWC

        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


class Food101ZeroShotDataset(Dataset):
    """
    Food-101 zero-shot dataset (torchvision).
    """

    def __init__(self, root, split="test", transform=None):
        self.transform = transform
        self.ds = Food101(
            root=root,
            split=split,
            download=False
        )
        self.classes = self.ds.classes

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
