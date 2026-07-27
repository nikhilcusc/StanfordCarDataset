import gradio as gr
from infer_saliency import infer_saliency

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
    saliency_map, prediction, confidence = infer_saliency(image, saliency_method)
    
    return saliency_map, prediction, confidence

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Saliency Map Visualization")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="pil")
                saliency_method = gr.Dropdown(
                    choices=["vanilla", "integrated_gradients", "lime", "shap"],
                    value="vanilla",
                    label="Saliency Method"
                )
                submit_button = gr.Button("Generate Saliency Map")
            
            with gr.Column():
                saliency_output = gr.Image(label="Saliency Map")
                
        with gr.Row():
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            confidence_output = gr.Number(label="Confidence Score", interactive=False)
        
        submit_button.click(
            fn=process_image,
            inputs=[image_input, saliency_method],
            outputs=[saliency_output, prediction_output, confidence_output]
        )
    
    demo.launch(share=True)

if __name__ == "__main__":
    main()
