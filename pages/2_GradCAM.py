"""
pages/2_GradCAM.py
Grad-CAM attention heatmap visualization for uploaded MRI images.
"""

import streamlit as st
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui_helpers    import (inject_page_css, render_sidebar_nav, model_selector,
                                  render_prediction_result, disclaimer_box)
from utils.preprocessing import load_image_from_upload, validate_image, preprocess_for_display
from utils.model_loader  import load_pytorch_model, predict_pytorch, load_tensorflow_model
from utils.gradcam       import (ViTGradCAM, compute_tf_gradcam,
                                  overlay_heatmap, get_attention_regions)

# ── Setup ──────────────────────────────────────────────────────────────
inject_page_css()
render_sidebar_nav()
model_type, model_path = model_selector()

# ── Page header ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔥 Grad-CAM Attention Heatmap</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Visual explanation of which brain regions the model focuses on when making its prediction</div>', unsafe_allow_html=True)

# ── Explanation card ───────────────────────────────────────────────────
with st.expander("ℹ️ What is Grad-CAM?", expanded=False):
    st.markdown("""
    **Gradient-weighted Class Activation Mapping (Grad-CAM)** is an explainability technique
    that produces a heatmap highlighting the image regions most important for the model's prediction.

    **How it works for ViT-B16:**
    1. The MRI image is tokenized into **196 patches** (14×14 grid)
    2. A forward pass produces the prediction
    3. Gradients of the predicted class score are computed with respect to the
       last transformer block's activations
    4. Patch tokens are **weighted by mean gradient magnitude**
    5. The 14×14 importance map is **upsampled to 224×224** and colorized

    **Colors:** 🔵 Blue = low attention · 🟡 Yellow = medium · 🔴 Red = high attention

    **Clinical value:** Clinicians can verify the model is attending to the tumor mass,
    not background noise or scanner artifacts.
    """)

st.markdown("---")

# ── Load model ─────────────────────────────────────────────────────────
model_loaded = False
model = None
device = None

if not os.path.exists(model_path):
    st.warning(f"⚠️ Model not found at `{model_path}`. Please check path in sidebar.")
else:
    with st.spinner("Loading model..."):
        if model_type == "pytorch":
            model, device = load_pytorch_model(model_path)
            model_loaded  = model is not None
        else:
            model  = load_tensorflow_model(model_path)
            model_loaded = model is not None

# ── Image input ────────────────────────────────────────────────────────
# Check if image was passed from Prediction page via session state
from_prediction = False
if "last_image" in st.session_state and st.session_state["last_image"] is not None:
    from_prediction = True

st.markdown("### 📤 Select MRI Image")

tab1, tab2 = st.tabs(["📁 Upload New Image", "♻️ Use Image from Prediction Page"])

pil_image = None

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="gradcam_upload"
    )
    if uploaded_file:
        pil_image = load_image_from_upload(uploaded_file)
        if pil_image and not validate_image(pil_image):
            pil_image = None

with tab2:
    if from_prediction:
        st.success("✅ Image loaded from Prediction page.")
        if st.button("Use this image for Grad-CAM", use_container_width=True):
            pil_image = st.session_state["last_image"]
    else:
        st.info("ℹ️ Go to the Prediction page first, upload an image and run classification, then come back here.")

# ── Colormap selector ──────────────────────────────────────────────────
st.markdown("---")
cmap_options = {
    "JET (Red-Yellow-Blue)"  : "jet",
    "HOT (Black-Red-Yellow)" : "hot",
    "PLASMA"                 : "plasma",
    "VIRIDIS"                : "viridis",
}
cmap_choice = st.selectbox("Colormap", list(cmap_options.keys()))
alpha_val   = st.slider("Heatmap opacity", min_value=0.2, max_value=0.8,
                         value=0.45, step=0.05)

# ── Generate Grad-CAM ──────────────────────────────────────────────────
if pil_image and model_loaded:
    st.markdown("---")
    generate_btn = st.button("🔥 Generate Grad-CAM Heatmap", use_container_width=True)

    if generate_btn:
        with st.spinner("Computing Grad-CAM..."):
            try:
                import cv2

                cmap_cv = {
                    "jet"    : cv2.COLORMAP_JET,
                    "hot"    : cv2.COLORMAP_HOT,
                    "plasma" : cv2.COLORMAP_PLASMA,
                    "viridis": cv2.COLORMAP_VIRIDIS,
                }[cmap_options[cmap_choice]]

                if model_type == "pytorch":
                    from torchvision import transforms
                    import torch

                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                    ])
                    img_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)

                    # Get prediction
                    result = predict_pytorch(model, device, pil_image)

                    # Grad-CAM
                    gradcam_engine = ViTGradCAM(model)
                    heatmap        = gradcam_engine.generate(img_tensor, class_idx=result["pred_idx"])
                    gradcam_engine.remove_hooks()

                else:
                    import numpy as np
                    arr = np.array(pil_image.convert("RGB").resize((224,224)),
                                   dtype=np.float32) / 255.0
                    arr = (arr - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
                    arr = np.expand_dims(arr, 0)
                    result  = {}
                    heatmap = compute_tf_gradcam(model, arr)

                # Overlay
                overlay_img = overlay_heatmap(pil_image, heatmap,
                                               alpha=alpha_val, colormap=cmap_cv)

                # Store for display
                st.session_state["gradcam_heatmap"]  = heatmap
                st.session_state["gradcam_overlay"]  = overlay_img
                st.session_state["gradcam_original"] = pil_image
                st.session_state["gradcam_result"]   = result if result else {}

            except Exception as e:
                st.error(f"❌ Grad-CAM failed: {e}")
                st.exception(e)

elif pil_image is None:
    st.info("👆 Upload an image above to generate a Grad-CAM heatmap.")
elif not model_loaded:
    st.warning("⚠️ Model not loaded. Check the model path in the sidebar.")

# ── Display results ────────────────────────────────────────────────────
if "gradcam_overlay" in st.session_state:
    st.markdown("---")
    st.markdown("### 🎨 Grad-CAM Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original MRI**")
        orig = preprocess_for_display(st.session_state["gradcam_original"], 300)
        st.image(orig, use_container_width=True)

    with col2:
        hm = st.session_state["gradcam_heatmap"]
        st.markdown("**Attention Heatmap (Raw)**")
        import cv2
        heatmap_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        from PIL import Image
        st.image(Image.fromarray(heatmap_rgb), use_container_width=True)

    with col3:
        st.markdown("**Heatmap Overlay**")
        st.image(st.session_state["gradcam_overlay"], use_container_width=True)

    # Attention analysis
    st.markdown("---")
    st.markdown("### 🧠 Attention Analysis")

    hm = st.session_state["gradcam_heatmap"]
    region_desc = get_attention_regions(hm)
    peak_val    = float(hm.max())
    mean_val    = float(hm.mean())

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Peak Attention", f"{peak_val:.3f}")
    with a2:
        st.metric("Mean Attention", f"{mean_val:.3f}")
    with a3:
        focused_pct = float((hm > 0.5).sum()) / hm.size * 100
        st.metric("High-Attention Area", f"{focused_pct:.1f}% of image")

    st.markdown(f"""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:10px;
                padding:14px 18px;margin-top:8px'>
        <div style='font-size:13px;font-weight:600;color:#E6EDF3;margin-bottom:6px'>
            📍 Model Attention Region
        </div>
        <div style='font-size:13px;color:#8B949E'>{region_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # Prediction summary if available
    result = st.session_state.get("gradcam_result", {})
    if result:
        st.markdown("---")
        st.markdown("### 🎯 Prediction Summary")
        render_prediction_result(result)

    # Download button
    st.markdown("---")
    from io import BytesIO
    buf = BytesIO()
    st.session_state["gradcam_overlay"].save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Heatmap Overlay",
        data=buf.getvalue(),
        file_name="gradcam_overlay.png",
        mime="image/png",
        use_container_width=True
    )

disclaimer_box()
