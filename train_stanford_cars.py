from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

ImageFile.LOAD_TRUNCATED_IMAGES = True


REQUIRED_COLUMNS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "fname", "class_name"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainingArtifacts:
    checkpoint_path: Path
    confusion_matrix_path: Path
    confusion_matrix_plot_path: Path
    classification_report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ResNet50 on Stanford Cars.")
    parser.add_argument("--train-annotations", type=Path, required=True, help="CSV, Parquet, or pickle file containing labeled training rows.")
    parser.add_argument("--test-annotations", type=Path, default=None, help="Optional CSV, Parquet, or pickle file containing labeled test rows.")
    parser.add_argument("--train-images-dir", type=Path, default=Path("./archive/cars_train/cars_train"), help="Directory containing training images.")
    parser.add_argument("--test-images-dir", type=Path, default=Path("./archive/cars_test/cars_test"), help="Directory containing test images.")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Directory for checkpoints and evaluation artifacts.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=max(0, min(8, os.cpu_count() or 0)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported annotations format: {path.suffix}. Use CSV, Parquet, or pickle.")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    for column in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["fname"] = df["fname"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    df = df.dropna(subset=["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "fname", "class_name"])
    return df.reset_index(drop=True)


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


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


class StanfordCarsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        label_encoder: LabelEncoder,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.label_encoder = label_encoder
        self.transform = transform
        self.labels = self.label_encoder.transform(self.dataframe["class_name"].tolist()).astype(np.int64)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_path = resolve_image_path(self.image_root, row["fname"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            box = clamp_box(
                (
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                ),
                width=image.width,
                height=image.height,
            )
            cropped = image if box is None else image.crop(box)

        if self.transform is not None:
            image_tensor = self.transform(cropped)
        else:
            image_tensor = transforms.ToTensor()(cropped)

        return {
            "image": image_tensor,
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def build_model(num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    train_images_dir: Path,
    test_images_dir: Path,
    label_encoder: LabelEncoder,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = StanfordCarsDataset(train_df, train_images_dir, label_encoder, transform=train_transform)
    val_dataset = StanfordCarsDataset(val_df, train_images_dir, label_encoder, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_df is not None and not test_df.empty:
        test_dataset = StanfordCarsDataset(test_df, test_images_dir, label_encoder, transform=eval_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[GradScaler],
) -> Tuple[float, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    running_correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            predictions = outputs.argmax(dim=1)
            running_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)

    average_loss = running_loss / max(1, total)
    accuracy = running_correct / max(1, total)
    return average_loss, accuracy


class EarlyStopping:
    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_score = -float("inf")
        self.bad_epochs = 0

    def step(self, score: float, model: nn.Module, epoch: int, label_encoder: LabelEncoder) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            self.save_checkpoint(model, epoch, label_encoder)
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def save_checkpoint(self, model: nn.Module, epoch: int, label_encoder: LabelEncoder) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "label_classes": label_encoder.classes_,
            },
            self.checkpoint_path,
        )


def evaluate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_predictions)


def save_confusion_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
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
    plt.close()

    report_text = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    classification_report_path = output_dir / "classification_report.txt"
    classification_report_path.write_text(report_text, encoding="utf-8")

    print("\nConfusion matrix shape:", matrix.shape)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print(f"Confusion matrix plot saved to: {confusion_matrix_plot_path}")
    print(f"Classification report saved to: {classification_report_path}")
    print("\nClassification report:\n")
    print(report_text)

    return confusion_matrix_path, classification_report_path


def prepare_dataframes(
    train_path: Path,
    test_path: Optional[Path],
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], LabelEncoder]:
    train_df = load_dataframe(train_path)
    test_df = load_dataframe(test_path) if test_path is not None else None

    split_kwargs = dict(test_size=val_fraction, random_state=seed)
    try:
        train_split, val_split = train_test_split(train_df, stratify=train_df["class_name"], **split_kwargs)
    except ValueError:
        train_split, val_split = train_test_split(train_df, **split_kwargs)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["class_name"])

    if test_df is not None:
        unknown_test_classes = sorted(set(test_df["class_name"]) - set(label_encoder.classes_))
        if unknown_test_classes:
            raise ValueError(f"Test annotations contain classes not seen in training: {unknown_test_classes}")

    return train_split.reset_index(drop=True), val_split.reset_index(drop=True), None if test_df is None else test_df.reset_index(drop=True), label_encoder


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: Path,
    label_encoder: LabelEncoder,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    scaler = GradScaler(enabled=device.type == "cuda")
    early_stopper = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, scaler)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if early_stopper.step(val_acc, model, epoch, label_encoder):
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_resnet50_stanford_cars.pt"

    train_df, val_df, test_df, label_encoder = prepare_dataframes(
        args.train_annotations,
        args.test_annotations,
        args.val_fraction,
        args.seed,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_images_dir=args.train_images_dir,
        test_images_dir=args.test_images_dir,
        label_encoder=label_encoder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes=len(label_encoder.classes_))
    model = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        label_encoder=label_encoder,
    )

    artifacts = TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        confusion_matrix_path=args.output_dir / "confusion_matrix.csv",
        confusion_matrix_plot_path=args.output_dir / "confusion_matrix.png",
        classification_report_path=args.output_dir / "classification_report.txt",
    )

    if test_loader is None:
        print("No test annotations were provided, so test evaluation was skipped.")
        print(f"Best checkpoint saved to: {artifacts.checkpoint_path}")
        return

    y_true, y_pred = evaluate_predictions(model.to(device), test_loader, device)
    save_confusion_outputs(y_true, y_pred, label_encoder.classes_, args.output_dir)
    print(f"Best checkpoint saved to: {artifacts.checkpoint_path}")


if __name__ == "__main__":
    main()