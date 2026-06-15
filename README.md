# Stanford Cars Classification

This repository contains a PyTorch workflow for training and evaluating a ResNet50 model on the Stanford Cars dataset ([GitHub - jhpohovey/StanfordCars-Dataset](https://github.com/jhpohovey/StanfordCars-Dataset)). It includes notebook-based data exploration, training code, and saved model weights for inference.

## What's Inside

- `exploreData.ipynb` for quick dataset inspection and visualization.
- `train_stanford_cars.ipynb` and `train_stanford_cars.py` for model training.
- `infer.ipynb` for loading the trained checkpoint, running single-image inference, and generating saliency explanations.
- `outputs/` with the trained checkpoint, including `best_resnet50_stanford_cars.pt`.

## Inference Workflow

The inference notebook loads the ResNet50 checkpoint, applies the same ImageNet-style preprocessing used during training, and predicts a class for a sample Stanford Cars image. It also includes saliency-based explanations to help interpret what the model focuses on when making a prediction.

Key steps in `infer.ipynb`:

1. Load the model checkpoint from `outputs/best_resnet50_stanford_cars.pt`.
2. Read the Stanford Cars class metadata from `archive/cars_meta.mat`.
3. Preprocess a test image with resize, center crop, tensor conversion, and normalization.
4. Run inference to get the predicted class and confidence.
5. Compute and plot vanilla saliency, integrated gradients, and LIME maps.

## Results

The figures below are saved in the `figures/` folder and show two different cars for each explanation method.

### Vanilla Saliency

They measure how much a small change in each pixel would change the model’s output. The result is a saliency map showing pixel‑level importance.

| Car 1 | Car 2 |
| --- | --- |
| ![Vanilla Saliency 1](figures/vanilla.png) | ![Vanilla Saliency 2](figures/vanilla_2.png) |

Limitations:
1. Noisy and Hard to Interpret - Raw gradients often produce high‑frequency, noisy maps that don’t align with human‑interpretable features. This makes explanations visually unstable and difficult to trust.
1. Sensitive to Model Saturation - If the model is in a saturated region (e.g., ReLU outputs zero), gradients can vanish. This leads to misleading saliency maps that suggest nothing is important. - to fix use Integrated gradients!
1. Not Robust to Small Perturbations - Tiny changes in the input can drastically change the gradient map, showing that vanilla gradients lack stability.
1. Highlighting Edges Instead of Semantics - Gradients often emphasize edges or texture rather than the meaningful object parts the model actually uses.
1. Poor Localization - The method struggles to clearly identify which regions of the image drive the prediction, especially in complex scenes.

How to fix:
Vanilla Gradients - raw sensitivity map; simple but noisy
Use Guided Backpropagation - filters gradients to highlight edges; prettier but less faithful
Use SmoothGrad - averages gradients over noisy inputs; reduces noise and improves clarity

### Integrated Gradients

| Car 1 | Car 2 |
| --- | --- |
| ![Integrated Gradients 1](figures/IG.png) | ![Integrated Gradients 2](figures/IG_2.png) |

### LIME

LIME explains a single prediction of a black‑box model by:

1. Creating perturbed versions of the input image
1. Getting the model’s predictions for each perturbed sample
1. Weighting these samples based on how similar they are to the original
1. Fitting a simple, interpretable surrogate model (usually linear)
1. Using that surrogate to identify which parts of the image influenced the prediction

| Car 1 | Car 2 |
| --- | --- |
| ![LIME 1](figures/LIME.png) | ![LIME 2](figures/LIME_2.png) |

Limitations:
1. Hyperparameter sensitivity - Small changes (e.g., number of superpixels) can produce inconsistent explanations.
1. Out-of-distribution perturbations - Masking superpixels creates unrealistic images, causing the model to behave unpredictably.
1. Local linearity assumption - Vision models are highly nonlinear; a linear surrogate may poorly approximate local behavior.
1. Instability - Re-running LIME can yield different explanations due to randomness in segmentation and sampling.

### Discussion

Across both cars, all three methods mostly focus on the vehicle instead of the background, which suggests the model is learning from the car itself rather than the scene. Integrated gradients gives the most continuous attribution and usually traces the body, windows, and wheel areas more cleanly than the others.

Vanilla saliency is noticeably noisier and more pixel-level, so it is useful as a quick signal but less stable as an explanation. LIME produces larger, block-like regions and is easier to read at a glance, but the superpixel boundaries can make it feel more coarse than gradient-based methods. In these examples, LIME still captures the rough object outline and confirms that the model is relying on the car region, while integrated gradients provides the clearest fine-grained explanation.

The main takeaway is that the methods are consistent at a high level but differ in granularity: vanilla saliency is the noisiest, integrated gradients is the smoothest, and LIME is the most segmented.

## Quick Start

1. Open `train_stanford_cars.ipynb` to retrain or evaluate the model.
2. Open `infer.ipynb` to run prediction and saliency analysis on a sample image.
3. Load `outputs/best_resnet50_stanford_cars.pt` for inference.
4. Use `exploreData.ipynb` to inspect the dataset structure and sample images.
