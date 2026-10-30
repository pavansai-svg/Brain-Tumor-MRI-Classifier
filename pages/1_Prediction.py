"""
pages/1_Prediction.py
Upload an MRI image and get instant tumor classification.
"""

import streamlit as st
import sys
import os

# ── Path fix so utils imports work from pages/ ─────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui_helpers    import (inject_page_css, render_sidebar_nav, model_selector,
                                  render_prediction_result, render_probability_chart,
                                  render_confidence_gauge, show_image_info, disclaimer_box)
from utils.preprocessing import (load_image_from_upload, validate_image,
                                  preprocess_for_display, get_image_info)
from utils.model_loader  import (load_pytorch_model, predict_pytorch,
                                  load_tensorflow_model, predict_tensorflow)

# ── Setup ──────────────────────────────────────────────────────────────
inject_page_css()
render_sidebar_nav()
model_type, model_path = model_selector()

# ── Page header ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔬 MRI Tumor Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Upload a brain MRI scan to classify it into one of four categories with confidence scores</div>', unsafe_allow_html=True)

# ── Model status ───────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.warning(f"""
    ⚠️ Model file not found at `{model_path}`.

    **To fix this:**
    1. Copy your model file into the `models/` folder inside `brain_tumor_app/`
    2. Or update the path in the sidebar

    Expected files:
    - `models/best_model_pytorch.pth` (PyTorch)
    - `models/best_model_phase1.keras` (TensorFlow)
    """)
    model_loaded = False
else:
    with st.spinner(f"Loading {model_type} model..."):
        if model_type == "pytorch":
            model, device = load_pytorch_model(model_path)
            model_loaded  = model is not None
        else:
            model  = load_tensorflow_model(model_path)
            device = None
            model_loaded = model is not None

    if model_loaded:
        st.success(f"✅ {model_type.title()} model loaded from `{model_path}`")
    else:
        st.error("❌ Model failed to load. Check the path and file format.")

st.markdown("---")

# ── Upload section ─────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Drop your brain MRI here",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a brain MRI scan in JPG or PNG format. Image will be resized to 224×224 for inference."
    )

    if uploaded_file:
        pil_image = load_image_from_upload(uploaded_file)

        if pil_image and validate_image(pil_image):
            # Show uploaded image
            display_img = preprocess_for_display(pil_image, target_size=300)
            st.image(display_img, caption="Uploaded MRI Scan", use_container_width=True)

            # Image metadata
            info = get_image_info(pil_image)
            show_image_info(info)

            # Predict button
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button(
                "🔬 Classify This MRI",
                disabled=not model_loaded,
                use_container_width=True
            )
        else:
            pil_image   = None
            predict_btn = False
    else:
        pil_image   = None
        predict_btn = False

        # Placeholder
        st.markdown("""
        <div style='background:#161B22;border:2px dashed #30363D;border-radius:12px;
                    padding:48px;text-align:center;'>
            <div style='font-size:40px;opacity:0.4'>🧠</div>
            <div style='font-size:14px;color:#8B949E;margin-top:12px'>
                Upload a brain MRI image to begin classification
            </div>
            <div style='font-size:12px;color:#30363D;margin-top:6px'>
                JPG · PNG · BMP · TIFF supported
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Results section ────────────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Classification Results")

    if predict_btn and pil_image and model_loaded:
        with st.spinner("Running inference..."):
            if model_type == "pytorch":
                result = predict_pytorch(model, device, pil_image)
            else:
                result = predict_tensorflow(model, pil_image)

        if result:
            # Store in session state for GradCAM page
            st.session_state["last_result"]   = result
            st.session_state["last_image"]    = pil_image
            st.session_state["model_type"]    = model_type
            st.session_state["model_path"]    = model_path

            # Prediction box
            render_prediction_result(result)

            # Two charts side by side
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Class Probabilities**")
                fig = render_probability_chart(result["probabilities"])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**Confidence Gauge**")
                fig = render_confidence_gauge(result["confidence"])
                st.plotly_chart(fig, use_container_width=True)

            # Detailed breakdown
            st.markdown("#### 📋 Full Probability Breakdown")
            for label, prob in result["probabilities"].items():
                pct = prob * 100
                bar_color = "#4F8BF9" if label == result["class_label"] else "#30363D"
                st.markdown(f"""
                <div style='margin-bottom:8px'>
                    <div style='display:flex;justify-content:space-between;
                                font-size:12px;color:#8B949E;margin-bottom:3px'>
                        <span style='color:{"#E6EDF3" if label==result["class_label"] else "#8B949E"};
                                     font-weight:{"700" if label==result["class_label"] else "400"}'>{label}</span>
                        <span style='color:#E6EDF3;font-weight:600'>{pct:.2f}%</span>
                    </div>
                    <div style='height:6px;background:#1C2128;border-radius:3px;overflow:hidden'>
                        <div style='width:{pct}%;height:100%;background:{bar_color};border-radius:3px;
                                    transition:width 0.5s'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Link to GradCAM page
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 **Next step:** Go to the **🔥 Grad-CAM** page to see which brain regions influenced this prediction.")

    else:
        # Placeholder results
        st.markdown("""
        <div style='background:#161B22;border:1px solid #30363D;border-radius:12px;
                    padding:48px;text-align:center;'>
            <div style='font-size:40px;opacity:0.4'>📊</div>
            <div style='font-size:14px;color:#8B949E;margin-top:12px'>
                Results will appear here after classification
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Disclaimer ─────────────────────────────────────────────────────────
disclaimer_box()
