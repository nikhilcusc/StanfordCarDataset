import torch   
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import shap
from lime import lime_image
from skimage.segmentation import slic


# Vanilla Saliency logic
def compute_saliency(image_path, model, transform, device):
    orig_img, display_img = open_image_for_display(image_path)
    input_tensor = transform(orig_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    model.zero_grad()
    outputs = model(input_tensor)            # logits
    top_idx = outputs.argmax(dim=1).item()
    score = outputs[0, top_idx]
    score.backward()
    saliency = input_tensor.grad.data.abs().squeeze(0)   # (C, H, W)
    saliency, _ = torch.max(saliency, dim=0)           # (H, W)
    saliency = saliency.cpu().numpy()
    # Normalize to [0,1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return display_img, saliency, top_idx

def compute_integrated_gradients(
    image_path,
    model,
    transform,
    device,
    steps=50,
    baseline=None,
):
    """
    Returns:
        display_img: image for visualization
        attr_map: normalized (H, W) integrated gradients heatmap
        top_idx: predicted class index
    """
    orig_img, display_img = open_image_for_display(image_path)

    model.eval()

    # Input tensor in the same space the model sees during training
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Baseline should be in the same transformed space as input_tensor
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.to(device)
        if baseline.dim() == 3:
            baseline = baseline.unsqueeze(0)

    # Get predicted class
    with torch.no_grad():
        logits = model(input_tensor)
        top_idx = logits.argmax(dim=1).item()

    # Integrated gradients accumulation
    total_gradients = torch.zeros_like(input_tensor)

    for alpha in torch.linspace(0, 1, steps, device=device):
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        outputs = model(interpolated)
        score = outputs[0, top_idx]

        grads = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            retain_graph=False,
            create_graph=False,
        )[0]

        total_gradients += grads.detach()

    avg_gradients = total_gradients / steps
    attributions = (input_tensor - baseline) * avg_gradients

    # Collapse channels -> single heatmap
    attr_map = attributions.squeeze(0).abs()
    attr_map, _ = torch.max(attr_map, dim=0)  # (H, W)

    attr_map = attr_map.cpu().numpy()

    # Normalize to [0, 1]
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

    return display_img, attr_map, top_idx

# Visualization: original and saliency heatmap side-by-side
def open_image_for_display(path):
    img = Image.open(path).convert("RGB")
    img_disp = img.resize((224, 224))
    return img, img_disp

def plot_saliency(display_img, saliency, cmap="hot", title="Saliency Map"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(display_img)
    axes[0].axis("off")
    axes[0].set_title("Original")
    im = axes[1].imshow(saliency, cmap=cmap)
    axes[1].axis("off")
    axes[1].set_title(title)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def _forward_logits(model, x):
    """
    Handles models that return logits directly or tuples/lists.
    """
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _to_pil(img):
    """
    Convert numpy / PIL / tensor-like image to PIL RGB.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            arr = np.clip(img, 0, 255).astype(np.uint8)
        else:
            arr = img
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _predict_proba_from_images(images, model, transform, device):
    """
    LIME classifier_fn: takes a list/array of images and returns class probabilities.
    """
    model.eval()
    tensors = []

    for img in images:
        pil_img = _to_pil(img)
        tensors.append(transform(pil_img))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits = _forward_logits(model, batch)
        probs = torch.softmax(logits, dim=1)

    return probs.detach().cpu().numpy()


def _normalize_heatmap(arr):
    arr = arr.astype(np.float32)
    arr = np.abs(arr)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def compute_lime_saliency(
    image_path,
    model,
    transform,
    device,
    num_samples=1000,
    num_features=10,
    top_labels=1,
    n_segments=100,
    compactness=10,
    sigma=1,
):
    """
    Returns:
        display_img: image for plotting
        heatmap: normalized HxW LIME explanation
        top_idx: predicted class index
    """
    orig_img, display_img = open_image_for_display(image_path)
    model.eval()

    # Predict top class
    input_tensor = transform(orig_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = _forward_logits(model, input_tensor)
        top_idx = logits.argmax(dim=1).item()

    explainer = lime_image.LimeImageExplainer()

    def classifier_fn(images):
        return _predict_proba_from_images(images, model, transform, device)

    image_np = np.array(display_img.convert("RGB") if isinstance(display_img, Image.Image) else display_img)

    explanation = explainer.explain_instance(
        image_np,
        classifier_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: slic(
            x,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0,
        ),
    )

    segments = explanation.segments
    local_exp = dict(explanation.local_exp.get(top_idx, []))

    heatmap = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in local_exp.items():
        heatmap[segments == seg_id] = weight

    heatmap = _normalize_heatmap(heatmap)
    return display_img, heatmap, top_idx


def compute_shap_saliency(
    image_path,
    model,
    transform,
    device,
    background_paths=None,
    background_images=None,
    background_size=8,
    nsamples=50,
):
    """
    Returns:
        display_img: image for plotting
        heatmap: normalized HxW SHAP explanation
        top_idx: predicted class index

    Notes:
    - SHAP works best with a small representative background set.
    - If no background is provided, this falls back to zeros.
    """
    orig_img, display_img = open_image_for_display(image_path)
    model.eval()

    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = _forward_logits(model, input_tensor)
        top_idx = logits.argmax(dim=1).item()

    # Build background batch
    bg_tensors = []

    if background_images is not None:
        for img in background_images[:background_size]:
            bg_tensors.append(transform(_to_pil(img)))
    elif background_paths is not None:
        for p in background_paths[:background_size]:
            bg_orig, _ = open_image_for_display(p)
            bg_tensors.append(transform(bg_orig))
    else:
        # Simple fallback; better to provide a small background set if possible.
        bg_tensors = [torch.zeros_like(input_tensor.squeeze(0)) for _ in range(background_size)]

    background = torch.stack(bg_tensors).to(device)

    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(input_tensor, nsamples=nsamples)

    # Handle common return formats
    if isinstance(shap_values, list):
        sv = shap_values[top_idx]
        # expected shape: (1, C, H, W)
        if isinstance(sv, np.ndarray):
            sv = sv[0]
    else:
        sv = shap_values
        # possible shapes: (1, C, H, W) or (1, num_classes, C, H, W)
        sv = np.array(sv)
        if sv.ndim == 5:
            sv = sv[0, top_idx]
        elif sv.ndim == 4:
            sv = sv[0]

    # Collapse channels to 2D heatmap
    if isinstance(sv, torch.Tensor):
        sv = sv.detach().cpu().numpy()

    heatmap = np.max(np.abs(sv), axis=0)
    heatmap = _normalize_heatmap(heatmap)

    return display_img, heatmap, top_idx