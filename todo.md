# TODO

## Documentation
- [ ] Polish language in README to introduce Explainability Explorer.
- [ ] Add a short overview of the Explainability Explorer workflow and what users can do with it.

## App
- [x] Build the Explainability Explorer in Gradio.
- [ ] Design a clean flow for loading an image, running inference, and displaying explanations.
- [x] Make the interface usable for quick experimentation and comparison.

## Models
- [ ] Include different models for inference and comparison.
- [ ] Add a model selector in the Gradio UI.
- [ ] Verify the selected model works with the same preprocessing pipeline.

## Saliency Methods
- [x] Add more saliency methods beyond the current set.
- [ ] Standardize outputs so each method can be displayed consistently.
- [ ] Confirm each method can run on the same input image.

## Comparison
- [ ] Allow the user to compare 2 methods side by side.
- [ ] Support matched inputs so both methods run on the same image and model output.
- [ ] Show explanations and metrics together for easier interpretation.

## Metrics
- [ ] Add model confidence.
- [ ] Show top-5 predictions.
- [ ] Show class probabilities.
- [ ] Add faithfulness score.
- [ ] Add deletion metric.
- [ ] Add insertion metric.

## Follow-Up
- [ ] Test the full end-to-end flow on a few sample car images.
- [ ] Save a few example outputs for the README or project notes.
