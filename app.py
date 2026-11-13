"""
Streamlit app (Phase 7): Brain Tumor MRI Classification UI.

What it does:
  - lets the user upload a JPG/PNG brain MRI image
  - shows the original image and the preprocessed (model input) image
  - shows the predicted class + confidence + probabilities for all
    4 classes as a bar chart
  - shows the Grad-CAM heatmap overlay (same engine as Phase 6)
  - shows a rough "focus region" box around the highest-attention area
    (a visual aid - NOT an exact tumor boundary)
  - warns when confidence is below 70% (uncertain prediction)
  - shows the model's test metrics in the sidebar
  - carries one MUTED amber disclaimer banner (the model is not
    clinically validated, so the app must never look like a
    diagnostic device - but the styling stays calm, not an alarm)

The model is loaded ONCE with st.cache_resource and reused on every
rerun/upload. Errors (invalid file, missing model) are handled
gracefully with messages, never with a crash.

Run from the project root:
    streamlit run app.py
"""

import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    OUTPUTS_DIR,
    PIPELINE,
)
from src.gradcam import GradCAM, focus_region_box, overlay_heatmap
from src.model import build_model
from src.preprocessing import normalize_for_model, preprocess_for_model

EVAL_SUMMARY_JSON = OUTPUTS_DIR / "evaluation_summary.json"
CONFIDENCE_THRESHOLD = 0.70   # below this -> "uncertain prediction" warning
NOTUMOR_IDX = CLASS_NAMES.index("notumor")

# --- Palette banners (custom HTML, not st.error/st.warning) ------------
# The project rule: never imply the model is clinically validated. The
# note appears as a MUTED amber banner under the title - visible but calm.
DISCLAIMER_BANNER = ('<div class="mt-banner disclaimer">'
                     'Research prototype — not clinically validated. '
                     'Not a medical device.</div>')

# Shared CSS for the banners. Amber #F59E0B on translucent amber panels;
# the disclaimer variant is deliberately more muted than the warning.
THEME_CSS = """
<style>
    .mt-banner {
        padding: 0.7rem 1rem;
        border-radius: 0.5rem;
        margin: 0.3rem 0 1rem 0;
        font-size: 0.95rem;
    }
    /* Disclaimer: muted amber */
    .mt-banner.disclaimer {
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #F5C983;
    }
    /* Low-confidence warning: amber */
    .mt-banner.warning {
        background: rgba(245, 158, 11, 0.14);
        border: 1px solid #F59E0B;
        color: #F59E0B;
        font-size: 1rem;
    }
</style>
"""

st.set_page_config(page_title="Brain Tumor Classifier",
                   page_icon="🩺", layout="wide")


# ----------------------------------------------------------------------
# Resource loading (cached: happens once per server session)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    """
    Load the trained model + Grad-CAM engine ONCE.

    Returns:
        (model, cam_engine, pipeline_name)
    """
    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu",
                            weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    pipeline = checkpoint.get("pipeline", PIPELINE)
    cam = GradCAM(model)          # hook stays attached for the session
    return model, cam, pipeline


def load_metrics() -> dict | None:
    """Test metrics produced by Phase 5 (None if not evaluated yet)."""
    if EVAL_SUMMARY_JSON.exists():
        return json.loads(EVAL_SUMMARY_JSON.read_text())
    return None


# ----------------------------------------------------------------------
# Inference logic (pure functions, testable without the UI)
# ----------------------------------------------------------------------

def read_upload_as_bgr(uploaded_file) -> np.ndarray | None:
    """
    Decode an uploaded file (bytes) into an OpenCV BGR image.
    Returns None for invalid files - handled by the caller.
    """
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    # IMREAD_COLOR always returns 3 channels (grayscale files are replicated)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def predict(image_bgr: np.ndarray, model, cam: GradCAM, pipeline: str):
    """
    Full inference for one image.

    Returns:
        (rgb, probs, pred_idx, confidence, overlay, focus) where
        rgb      = preprocessed model input (uint8 RGB, 224x224)
        probs    = numpy array of 4 class probabilities
        overlay  = Grad-CAM heatmap blended onto the preprocessed image
        focus    = image with a box around the highest-attention region;
                   for 'notumor' predictions the unmodified image is
                   returned (a "tumor location" box would be misleading)
    """
    rgb = preprocess_for_model(image_bgr, pipeline)
    norm = normalize_for_model(rgb)
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():                     # classification: no gradients
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())

    # Grad-CAM does its own (gradient-enabled) forward pass.
    heatmap, _ = cam.generate(tensor, class_idx=pred_idx)
    overlay = overlay_heatmap(rgb, heatmap)
    if pred_idx != NOTUMOR_IDX:
        focus = focus_region_box(rgb, heatmap)
    else:
        # No tumor predicted -> there is no "tumor location" to show.
        # Drawing the highest-attention box here would just highlight a
        # random-looking area, so the plain image is returned instead.
        focus = rgb.copy()

    return rgb, probs.cpu().numpy(), pred_idx, confidence, overlay, focus


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

def main() -> None:
    # --- Header ----------------------------------------------------------
    st.title("🩺 Brain Tumor MRI Classification")
    # Theme CSS + the muted amber disclaimer banner (always visible).
    st.markdown(THEME_CSS + DISCLAIMER_BANNER, unsafe_allow_html=True)

    # --- Load model (handle missing/corrupt checkpoint gracefully) ------
    try:
        model, cam, pipeline = load_resources()
    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
        st.info("Train the model first:  `python -m src.train`")
        st.stop()

    # --- Sidebar: model info + test metrics ------------------------------
    st.sidebar.header("Model")
    st.sidebar.write(f"**Architecture:** ResNet50 (ImageNet-pretrained, "
                     f"fine-tuned, 2 stages)")
    st.sidebar.write(f"**Preprocessing pipeline:** `{pipeline}`")
    st.sidebar.write(f"**Device:** `{DEVICE.type}`")

    st.sidebar.subheader("Test-set metrics (held-out images)")
    metrics = load_metrics()
    if metrics:
        col_a, col_b = st.sidebar.columns(2)
        col_a.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        col_b.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")

        per_class_df = pd.DataFrame(metrics["per_class"]).T
        st.sidebar.dataframe(per_class_df, width="stretch")

        ts = metrics["tumor_safety"]
        st.sidebar.info(
            f"**Tumor safety:** 'any tumor' recall {ts['tumor_recall']:.3f} "
            f"({ts['n_missed_as_notumor']} of {ts['n_tumor_images']} tumor "
            f"images were missed as 'notumor').")

        with st.sidebar.expander("About these metrics"):
            st.caption(metrics["leakage_note"])
    else:
        st.sidebar.warning("No evaluation summary found. Run "
                           "`python -m src.evaluate` to generate it.")

    # --- Upload -----------------------------------------------------------
    st.subheader("Upload a brain MRI image (JPG / PNG)")
    uploaded = st.file_uploader(
        "MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded is None:
        st.info("Upload an image to see the preprocessed version, the "
                "prediction, the Grad-CAM heatmap and the focus region.")
        st.stop()

    image_bgr = read_upload_as_bgr(uploaded)
    if image_bgr is None:
        st.error("Could not read this file as an image. Please upload a "
                 "valid JPG or PNG brain MRI scan.")
        st.stop()

    # --- Predict + display ------------------------------------------------
    with st.spinner("Analyzing the scan..."):
        try:
            rgb, probs, pred_idx, confidence, overlay, focus = predict(
                image_bgr, model, cam, pipeline)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    st.subheader(f"Predicted class: **{CLASS_NAMES[pred_idx]}** "
                 f"— confidence **{confidence * 100:.1f}%**")

    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown(
            f'<div class="mt-banner warning">⚠️ <b>Uncertain prediction.</b> '
            f'Confidence is below {CONFIDENCE_THRESHOLD * 100:.0f}% - treat '
            f'this class label with caution and do not rely on it.</div>',
            unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
               caption=f"Original ({uploaded.name})", width="stretch")
    col2.image(rgb, caption="Preprocessed model input (224×224)",
               width="stretch")
    col3.image(overlay, caption="Grad-CAM overlay (warm = influential)",
               width="stretch")
    if pred_idx == NOTUMOR_IDX:
        col4.image(focus,
                   caption="No focus box (predicted: no tumor)",
                   width="stretch")
    else:
        col4.image(focus,
                   caption="Focus region (rough) - where the model "
                   "looked", width="stretch")

    st.caption("How to read the heatmap: warm colors mark regions that "
               "influenced the prediction, and the box marks the area "
               "with the strongest attention. Neither is an exact tumor "
               "boundary - Grad-CAM is NOT tumor segmentation.")

    st.subheader("Class probabilities")
    prob_df = pd.DataFrame({"probability": probs}, index=CLASS_NAMES).T
    st.bar_chart(prob_df)

    with st.expander("Show raw probabilities"):
        st.dataframe(prob_df.T.style.format("{:.4f}"), width="stretch")


if __name__ == "__main__":
    main()
