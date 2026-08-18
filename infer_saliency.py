from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from saliencyDefs import (
    compute_gradcam,
    compute_guided_backprop,
    compute_integrated_gradients,
    compute_lime_saliency,
    compute_saliency,
    compute_shap_saliency,
)
from utils import load_annotation_file


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "outputs" / "best_resnet50_stanford_cars.pt"
DEFAULT_META_FILE = BASE_DIR / "archive" / "cars_meta.mat"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "saliency"


def build_resnet50(num_classes: int) -> nn.Module:
    try:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        model = resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def infer_num_classes_from_state_dict(state_dict, fallback_num_classes: int) -> int:
    fc_weight = state_dict.get("fc.weight") if isinstance(state_dict, dict) else None
    if isinstance(fc_weight, torch.Tensor) and fc_weight.ndim == 2:
        return fc_weight.shape[0]
    return fallback_num_classes


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    inferred_num_classes = infer_num_classes_from_state_dict(state_dict, model.fc.out_features)
    if model.fc.out_features != inferred_num_classes:
        model.fc = nn.Linear(model.fc.in_features, inferred_num_classes)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_preprocess_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    return outputs


def predict_image(image_path: Path, model: nn.Module, transform: transforms.Compose, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = forward_logits(model, input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    display_img = image.resize((224, 224))
    return display_img, predicted_index.item(), confidence.item()


def load_class_metadata(meta_file: Path) -> pd.DataFrame:
    metadata, loader_type = load_annotation_file(meta_file)

    if loader_type == "scipy":
        class_names = metadata["class_names"].squeeze()
        names = [cls.item() if hasattr(cls, "item") else str(cls) for cls in class_names]
    else:
        class_names = metadata["class_names"]
        names = []
        for ref in class_names[0]:
            obj = metadata[ref]
            values = obj[()]
            if values.dtype == np.uint16:
                names.append("".join(chr(code[0]) for code in values))
            else:
                names.append(values.item() if values.size == 1 else str(values))

    return pd.DataFrame({"class_id": np.arange(1, len(names) + 1), "class_name": names})


def resolve_method(method_name: str):
    normalized = method_name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "vanilla": "vanilla",
        "saliency": "vanilla",
        "vanilla_saliency": "vanilla",
        "integrated_gradients": "integrated_gradients",
        "ig": "integrated_gradients",
        "lime": "lime",
        "shap": "shap",
        "guided_backprop": "guided_backprop",
        "guided_backpropagation": "guided_backprop",
        "gbp": "guided_backprop",
        "gradcam": "gradcam",
        "grad_cam": "gradcam",
        "grad-cam": "gradcam",
    }

    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown method '{method_name}'. Choose one of: {valid}")

    return aliases[normalized]


def compute_saliency_map(image_path: Path, method_name: str, model: nn.Module, transform, device):
    method = resolve_method(method_name)

    if method == "vanilla":
        return compute_saliency(image_path, model, transform, device)
    if method == "integrated_gradients":
        return compute_integrated_gradients(image_path, model, transform, device)
    if method == "lime":
        return compute_lime_saliency(image_path, model, transform, device)
    if method == "shap":
        return compute_shap_saliency(image_path, model, transform, device)
    if method == "guided_backprop":
        return compute_guided_backprop(image_path, model, transform, device)
    if method == "gradcam":
        target_layer = model.layer4[-1]
        return compute_gradcam(image_path, model, target_layer, transform, device)

    raise ValueError(f"Unsupported saliency method: {method_name}")


def save_saliency_figure(display_img, saliency_map, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(display_img)
    axes[0].axis("off")
    axes[0].set_title("Original")

    image = axes[1].imshow(saliency_map, cmap="hot")
    axes[1].axis("off")
    axes[1].set_title(title)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def infer_saliency(image, saliency_method: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_file = DEFAULT_META_FILE
    checkpoint_path = DEFAULT_CHECKPOINT

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not meta_file.is_file():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    class_df = load_class_metadata(meta_file)
    model = build_resnet50(num_classes=len(class_df))
    model = load_model_weights(model, checkpoint_path, device)
    transform = get_preprocess_transform()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    temp_path = BASE_DIR / "outputs" / "saliency" / "_gradio_input.png"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(temp_path)

    display_img, predicted_idx, confidence = predict_image(temp_path, model, transform, device)
    saliency_display_img, saliency_map, _ = compute_saliency_map(temp_path, saliency_method, model, transform, device)

    class_name = class_df.iloc[predicted_idx]["class_name"]
    return saliency_map, class_name, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stanford Cars inference with a saliency explanation.")
    parser.add_argument("image_path", type=Path, help="Path to the input image")
    parser.add_argument(
        "method",
        type=str,
        help="Saliency method: vanilla, integrated_gradients, lime, shap, guided_backprop, or gradcam",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Model checkpoint path")
    parser.add_argument("--meta-file", type=Path, default=DEFAULT_META_FILE, help="Stanford Cars metadata file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.meta_file.is_file():
        raise FileNotFoundError(f"Metadata file not found: {args.meta_file}")

    class_df = load_class_metadata(args.meta_file)
    model = build_resnet50(num_classes=len(class_df))
    model = load_model_weights(model, args.checkpoint, device)
    transform = get_preprocess_transform()

    display_img, predicted_idx, confidence = predict_image(args.image_path, model, transform, device)
    saliency_display_img, saliency_map, saliency_predicted_idx = compute_saliency_map(
        args.image_path, args.method, model, transform, device
    )

    class_name = class_df.iloc[predicted_idx]["class_name"]
    method = resolve_method(args.method)
    saliency_title = f"{method.replace('_', ' ').title()} Saliency"
    output_path = args.output_dir / f"{args.image_path.stem}_{method}_saliency.png"
    save_saliency_figure(saliency_display_img, saliency_map, output_path, saliency_title)

    print(f"Predicted class index: {predicted_idx}")
    print(f"Predicted class name: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Saliency method: {method}")
    print(f"Saliency map saved to: {output_path}")

    if predicted_idx != saliency_predicted_idx:
        print(f"Warning: prediction index {predicted_idx} differs from saliency index {saliency_predicted_idx}")


if __name__ == "__main__":
    main()