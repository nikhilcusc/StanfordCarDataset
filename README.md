# Stanford Cars Classification

This repository contains a PyTorch workflow for training and evaluating a ResNet50 model on the Stanford Cars dataset. It includes notebook-based exploration, training code, dataset annotations, and saved model weights for inference.

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
5. Compute and plot both vanilla saliency and integrated gradients maps.

## Example Results

The figures below are saved in the `figures/` folder and summarize the explanation outputs for the same sample car image.

### Integrated Gradients

![Integrated Gradients](figures/IG.png)

### Vanilla Saliency

![Vanilla Saliency](figures/vanilla.png)

### Discussion

Both methods highlight the vehicle rather than the blank background, which is a good sign that the model is using image content instead of spurious context. The integrated gradients map is a little more structured and tends to concentrate on the car body, roofline, and wheel regions, while the vanilla saliency map is noisier and more diffuse.

That difference is expected: vanilla gradients can be sensitive to local pixel-level changes, so they often look speckled. Integrated gradients usually produce a smoother attribution map because they accumulate gradients along a path from a baseline to the input. In this example, integrated gradients gives the clearer explanation of why the model is leaning toward the predicted class.

## Quick Start

1. Open `train_stanford_cars.ipynb` to retrain or evaluate the model.
2. Open `infer.ipynb` to run prediction and saliency analysis on a sample image.
3. Load `outputs/best_resnet50_stanford_cars.pt` for inference.
4. Use `exploreData.ipynb` to inspect the dataset structure and sample images.
