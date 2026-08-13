"""Gradio Web Application for Deepfake Face Authenticity Detection & Layer-CAM Explainability.

Designed for native deployment on Hugging Face Spaces (sdk: gradio).
"""

# IMPORTANT FOR HUGGING FACE ZERO-GPU: spaces MUST be imported before torch!
try:
    import spaces
except Exception:
    spaces = None

import sys
import base64
import io
from pathlib import Path
from PIL import Image

import torch
import gradio as gr

# Ensure project root and backend directory are in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "webapp" / "backend"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Explicit module imports from webapp.backend.app
from webapp.backend.app.model import ModelRegistry
from webapp.backend.app import config, gradcam, preprocessing

# Initialize Model Registry
registry = ModelRegistry()

MODEL_LABELS = {
    "best": "Main Model (MobileNetV3 - CelebA/3-Stage Fine-tuned)",
    "cross_domain": "Cross-Domain Model (General non-face / multi-dataset)",
    "manipulations": "Manipulation Robustness Model (Trained under 11 distortions)",
    "noaug": "No-Augmentation Baseline Model",
    "siglip2_vit_finetuned_by_som": "siglip2_vit_finetuned_by_som",
}

_HF_SIGLIP_PIPE = None


def get_siglip_pipeline():
    global _HF_SIGLIP_PIPE
    if _HF_SIGLIP_PIPE is None:
        from transformers import pipeline
        _HF_SIGLIP_PIPE = pipeline(
            "image-classification", model="prithivMLmods/deepfake-detector-model-v1"
        )
    return _HF_SIGLIP_PIPE


def _predict_and_explain(image: Image.Image, model_choice: str):
    if image is None:
        return (
            "Please upload an image to analyze.",
            {},
            None,
            None,
            "No input image provided.",
        )

    # 1. Map readable model choice back to key
    model_key = "best"
    for k, label in MODEL_LABELS.items():
        if label == model_choice or k == model_choice:
            model_key = k
            break

    # 2. Preprocess & Crop Face
    cropped_img, alignment_method = preprocessing.crop_and_align_face(image)

    # 3. Handle siglip2_vit_finetuned_by_som vs PyTorch MobileNetV3 models
    if model_key == "siglip2_vit_finetuned_by_som":
        pipe = get_siglip_pipeline()
        outputs = pipe(cropped_img)
        
        real_pct = 0.0
        fake_pct = 0.0
        for item in outputs:
            lbl = item["label"].capitalize()
            score = item["score"] * 100.0
            if "Real" in lbl or "Genuine" in lbl:
                real_pct = score
            elif "Fake" in lbl or "Synthetic" in lbl:
                fake_pct = score

        # If labels don't explicitly say real/fake, assign highest score
        if real_pct == 0.0 and fake_pct == 0.0 and len(outputs) >= 2:
            fake_pct = outputs[0]["score"] * 100.0
            real_pct = outputs[1]["score"] * 100.0

        verdict = "Fake" if fake_pct >= real_pct else "Real"

        # Use best backbone model for Layer-CAM visualization
        model_obj = registry.get("best")
        if model_obj is not None:
            cam_res = gradcam.gradcam_overlay(
                model_obj,
                cropped_img,
                preprocessing.val_transform,
                registry.device,
                alpha=0.65,
            )
            heatmap_bytes = base64.b64decode(cam_res["heatmap_b64"])
            overlay_bytes = base64.b64decode(cam_res["overlay_b64"])
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
            overlay_img = Image.open(io.BytesIO(overlay_bytes))
        else:
            heatmap_img = cropped_img
            overlay_img = cropped_img

        model_display_name = "siglip2_vit_finetuned_by_som"
    else:
        model_obj = registry.get(model_key)
        if model_obj is None:
            return (
                f"Error: Model '{model_key}' is not loaded.",
                {},
                None,
                None,
                "Model unavailable.",
            )

        result = gradcam.gradcam_overlay(
            model_obj,
            cropped_img,
            preprocessing.val_transform,
            registry.device,
            alpha=0.65,
        )

        pred_info = result["prediction"]
        verdict = pred_info["label"]
        real_pct = pred_info["real_pct"]
        fake_pct = pred_info["fake_pct"]

        heatmap_bytes = base64.b64decode(result["heatmap_b64"])
        overlay_bytes = base64.b64decode(result["overlay_b64"])

        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
        overlay_img = Image.open(io.BytesIO(overlay_bytes))
        model_display_name = MODEL_LABELS.get(model_key, model_key)

    confidence_label = f"## Verdict: **{verdict.upper()}**\n\n- **Real Confidence**: `{real_pct:.2f}%`\n- **Fake Confidence**: `{fake_pct:.2f}%`"

    prob_dict = {
        "Real": real_pct / 100.0,
        "Fake": fake_pct / 100.0,
    }

    metadata_text = f"""**Model In Use**: `{model_display_name}`
**Face Alignment**: `{alignment_method}`
**Input Image Resolution**: `{image.width}x{image.height}` $\\rightarrow$ Cropped to `{config.IMG_SIZE}x{config.IMG_SIZE}`
**Layer-CAM Target Layers**: `features[11]` + `features[-1]`
"""

    return confidence_label, prob_dict, overlay_img, heatmap_img, metadata_text


# Zero-GPU decoration if running on HF Spaces ZeroGPU hardware
if spaces is not None and hasattr(spaces, "GPU"):
    analyze_image = spaces.GPU(_predict_and_explain)
else:
    analyze_image = _predict_and_explain


# Build Gradio Blocks UI
with gr.Blocks(title="Deepfake Face Detection & Explainability") as demo:
    gr.Markdown(
        """
        # 🔍 Deepfake Face Authenticity & Layer-CAM Explainability System
        Upload a facial photograph to evaluate its authenticity. The system runs real-time face detection, MobileNetV3 / SigLIP Vision Transformer classification, and **Layer-CAM spatial saliency mapping** to pinpoint forgery evidence.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Facial Image")
            model_selector = gr.Dropdown(
                choices=list(MODEL_LABELS.values()),
                value=MODEL_LABELS["best"],
                label="Select Detection Model",
            )
            submit_btn = gr.Button("🔍 Analyze Authenticity", variant="primary")

        with gr.Column(scale=1):
            verdict_markdown = gr.Markdown("### Verdict will appear here")
            confidence_chart = gr.Label(label="Class Probabilities", num_top_classes=2)
            metadata_box = gr.Markdown("### Analysis Details")

    gr.Markdown("---")
    gr.Markdown("## 🧬 Layer-CAM Explainability (Saliency Evidence)")

    with gr.Row():
        with gr.Column():
            overlay_output = gr.Image(
                type="pil", label="Adaptive Turbo Layer-CAM Overlay"
            )
            gr.Markdown(
                "*Highlights forgery regions (bright turbo colors) on top of the original face with adaptive transparency.*"
            )
        with gr.Column():
            heatmap_output = gr.Image(
                type="pil", label="Standalone Layer-CAM Heatmap"
            )
            gr.Markdown(
                "*Perceptually uniform Turbo saliency map showing exact pixel activation intensity.*"
            )

    submit_btn.click(
        fn=analyze_image,
        inputs=[input_img, model_selector],
        outputs=[
            verdict_markdown,
            confidence_chart,
            overlay_output,
            heatmap_output,
            metadata_box,
        ],
    )

# For Hugging Face Spaces (app.py)
if __name__ == "__main__":
    demo.launch()
