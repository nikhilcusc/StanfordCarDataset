import torch   
import matplotlib.pyplot as plt
from PIL import Image


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