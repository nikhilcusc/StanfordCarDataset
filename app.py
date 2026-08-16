import gc
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from infer_saliency import infer_saliency

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def release_memory():
    """
    Explicitly release cached memory held by PyTorch and the garbage collector.

    Each Gradio click loads a fresh ResNet50 model inside infer_saliency, so
    freeing references and clearing the cache after every request prevents
    GPU/CPU memory from accumulating across repeated interactions.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_image(image, saliency_method):
    """
    Process image with selected saliency method

    Args:
        image: PIL Image or numpy array
        saliency_method: str, selected saliency method

    Returns:
        saliency_map: numpy array, the saliency map
        prediction: str, the predicted class
        confidence: float, the confidence score
    """
    try:
        saliency_map, prediction, confidence = infer_saliency(image, saliency_method)

        # Convert to a plain numpy array so no torch tensors remain referenced
        # after the function returns (helps free memory sooner).
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.detach().cpu().numpy()
        elif isinstance(saliency_map, np.ndarray):
            saliency_map = np.asarray(saliency_map)

        return saliency_map, prediction, confidence
    except Exception as exc:
        raise exc
    finally:
        # Always release memory, even when inference raises an error.
        release_memory()


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Saliency Map Visualization")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="pil")
                saliency_method = gr.Dropdown(
                    choices=[
                        "vanilla",
                        "integrated_gradients",
                        "lime",
                        "shap",
                        "guided_backprop",
                        "gradcam",
                    ],
                    value="vanilla",
                    label="Saliency Method"
                )
                ig_note = gr.Markdown(
                    "**Note:** Running only 50 iterations for Integrated Gradients.",
                    visible=False,
                )
                submit_button = gr.Button("Generate Saliency Map")

            with gr.Column():
                saliency_output = gr.Image(label="Saliency Map")

        with gr.Row():
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            confidence_output = gr.Number(label="Confidence Score", interactive=False)

        gr.Examples(
            examples=[
                [str(img), "vanilla"]
                for img in sorted(EXAMPLES_DIR.glob("*"))
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            inputs=[image_input, saliency_method],
            outputs=[saliency_output, prediction_output, confidence_output],
            fn=process_image,
            label="Example Images",
            run_on_click=True,
        )

        def toggle_ig_note(method):
            return gr.update(visible=(method == "integrated_gradients"))

        saliency_method.change(
            fn=toggle_ig_note,
            inputs=saliency_method,
            outputs=ig_note,
        )

        submit_button.click(
            fn=process_image,
            inputs=[image_input, saliency_method],
            outputs=[saliency_output, prediction_output, confidence_output]
        )

    demo.launch(share=True,
                server_port=7862)


if __name__ == "__main__":
    main()
