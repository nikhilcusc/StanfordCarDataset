from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image, ImageFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from sklearn.metrics import classification_report, confusion_matrix
import scipy.io
import h5py
#from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "fname", "class_name"]

def load_annotation_file(filepath: Path):
    """Load a MATLAB .mat annotation file using scipy first, then h5py for v7.3 files."""
    if not filepath.exists():
        raise FileNotFoundError(f"Annotation file not found: {filepath}")

    try:
        mat_data = scipy.io.loadmat(filepath)
        return mat_data, "scipy"
    except NotImplementedError:
        mat_data = h5py.File(filepath, "r")
        return mat_data, "h5py"

def extract_to_dataframe(mat, loader_type: str) -> pd.DataFrame:
    records = []

    if loader_type == "scipy":
        anno_keys = [k for k in mat.keys() if "anno" in k.lower()]
        anno_key = anno_keys[-1] if anno_keys else None
        if anno_key and anno_key in mat:
            arr = mat[anno_key]
            if arr.dtype.names:
                for row in arr[0]:
                    record = {
                        name: row[name].item() if row[name].size == 1 else row[name]
                        for name in arr.dtype.names
                    }
                    for key, value in list(record.items()):
                        if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
                            record[key] = str(value[0]) if value.size > 0 else ""
                        elif isinstance(value, bytes):
                            record[key] = value.decode("utf-8")
                    records.append(record)
    else:
        anno_keys = [k for k in mat.keys() if "anno" in k.lower()]
        anno_key = anno_keys[-1] if anno_keys else None
        if anno_key and anno_key in mat:
            for ref in mat[anno_key][0]:
                obj = mat[ref]
                record = {}
                for key in obj.keys():
                    value = obj[key][()]
                    if value.dtype == np.uint16:
                        record[key] = "".join(chr(c[0]) for c in value)
                    else:
                        record[key] = value.item() if value.size == 1 else value
                records.append(record)

    return pd.DataFrame(records)


def clamp_box(box: Tuple[float, float, float, float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = int(round(max(0, min(x1, width - 1))))
    y1 = int(round(max(0, min(y1, height - 1))))
    x2 = int(round(max(1, min(x2, width))))
    y2 = int(round(max(1, min(y2, height))))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def resolve_image_path(image_root: Path, fname: str) -> Path:
    candidate = Path(fname)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    joined = image_root / candidate
    if joined.exists():
        return joined

    if candidate.exists():
        return candidate

    return joined

def build_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform

class StanfordCarsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: Path, label_encoder: LabelEncoder, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.label_encoder = label_encoder
        self.transform = transform
        self.labels = self.label_encoder.transform(self.dataframe["class_name"].tolist()).astype(np.int64)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = resolve_image_path(self.image_root, row["fname"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            box = clamp_box(
                (float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])),
                width=image.width,
                height=image.height,
            )
            cropped = image if box is None else image.crop(box)

        if self.transform is not None:
            image_tensor = self.transform(cropped)
        else:
            image_tensor = transforms.ToTensor()(cropped)

        return {"image": image_tensor, "label": torch.tensor(self.labels[index], dtype=torch.long)}

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["fname"] = df["fname"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    return df

def evaluate_predictions(model: nn.Module, dataloader: DataLoader, 
                         DEVICE):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in (dataloader):
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_predictions)

def save_confusion_outputs(y_true: np.ndarray, y_pred: np.ndarray, label_names: np.ndarray, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    matrix_df = pd.DataFrame(matrix, index=label_names, columns=label_names)

    confusion_matrix_path = output_dir / "confusion_matrix.csv"
    matrix_df.to_csv(confusion_matrix_path)

    confusion_matrix_plot_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(max(12, len(label_names) * 0.18), max(10, len(label_names) * 0.18)))
    sns.heatmap(matrix_df, cmap="Blues", xticklabels=len(label_names) <= 30, yticklabels=len(label_names) <= 30)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(confusion_matrix_plot_path, dpi=200)
    plt.show()
    plt.close()

    report_text = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    classification_report_path = output_dir / "classification_report.txt"
    classification_report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    return confusion_matrix_path, confusion_matrix_plot_path, classification_report_path