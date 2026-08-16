import gc
from pathlib import Path
from PIL import Image

import streamlit as st
import numpy as np
import torch

from infer_saliency import infer_saliency

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Saliency Map Visualizer", 
    page_icon="🧠", 
    layout="wide"
)

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def release_memory():
    """
    Explicitly release cached memory held by PyTorch and the garbage collector.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_image(image_file):
    """Helper function to load PIL image from an uploaded file or path."""
    return Image.open(image_file).convert("RGB")


def main():
    st.title("Saliency Map Visualization")
    st.markdown("Upload an image or choose an example to generate visual explanations for the model's predictions.")

    # --- Sidebar for Inputs & Settings ---
    with st.sidebar:
        st.header("Settings")
        saliency_method = st.selectbox(
            "Saliency Method",
            options=[
                "vanilla",
                "integrated_gradients",
                "guided_backprop",
                "gradcam",
            ],
            index=0
        )

        if saliency_method == "integrated_gradients":
            st.info("**Note:** Running only 50 iterations for Integrated Gradients.")

        st.markdown("---")

        # Handle Example Images dynamically
        st.header("Examples")
        example_images = []
        if EXAMPLES_DIR.exists():
            example_images = sorted([
                img for img in EXAMPLES_DIR.glob("*") 
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])
        
        example_names = [img.name for img in example_images]
        selected_example = st.selectbox("Choose an example", ["None"] + example_names)

    # --- Main Layout Area ---
    # Create two columns for a side-by-side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload your own image...", type=["jpg", "jpeg", "png"])

    image_to_process = None

    # Logic to prioritize user upload over example selection
    if uploaded_file is not None:
        image_to_process = load_image(uploaded_file)
        with col1:
            st.image(image_to_process, use_column_width=True)
    elif selected_example != "None":
        example_path = EXAMPLES_DIR / selected_example
        image_to_process = load_image(example_path)
        with col1:
            st.image(image_to_process, caption=f"Example: {selected_example}", use_column_width=True)
    else:
        with col1:
            st.info("Please upload an image or select an example from the sidebar.")

    # --- Execution & Results ---
    st.write("") # Spacer
    
    # We span the button across the whole width using container width
    if st.button("Generate Saliency Map"):
        if image_to_process is None:
            st.warning("Please provide an image first!")
        else:
            # Spinner provides better visual feedback than Gradio's default loading text
            with st.spinner(f"Processing image with {saliency_method}..."):
                try:
                    saliency_map, prediction, confidence = infer_saliency(image_to_process, saliency_method)

                    # Detach tensors and convert to numpy
                    if isinstance(saliency_map, torch.Tensor):
                        saliency_map = saliency_map.detach().cpu().numpy()
                    elif isinstance(saliency_map, np.ndarray):
                        saliency_map = np.asarray(saliency_map)

                    with col2:
                        st.subheader("Saliency Map")
                        # clamp=True ensures float arrays (0.0 to 1.0) render correctly 
                        st.image(saliency_map, use_column_width=True, clamp=True)

                    st.markdown("---")
                    
                    # Use metric cards for a dashboard-style look
                    st.subheader("Model Output")
                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric(label="Predicted Class", value=str(prediction))
                    
                    # Format confidence as a percentage if it's a raw float
                    if isinstance(confidence, (float, np.floating)):
                        confidence_str = f"{confidence * 100:.2f}%" if confidence <= 1.0 else f"{confidence:.2f}"
                    else:
                        confidence_str = str(confidence)
                        
                    metric_col2.metric(label="Confidence Score", value=confidence_str)

                except Exception as exc:
                    st.error(f"An error occurred during inference: {exc}")
                finally:
                    # Clean up memory
                    release_memory()


if __name__ == "__main__":
    main()