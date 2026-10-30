"""
pages/4_About.py
Project details, tech stack, and deployment information.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ui_helpers import inject_page_css, render_sidebar_nav

inject_page_css()
render_sidebar_nav()

st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Brain Tumor MRI Classification using Vision Transformers with Grad-CAM Explainability</div>', unsafe_allow_html=True)

# ── Project summary ────────────────────────────────────────────────────
st.markdown("### 🎯 Project Overview")
st.markdown("""
<div style='background:#161B22;border:1px solid #30363D;border-radius:12px;padding:20px 24px;margin-bottom:16px'>
    <p style='color:#E6EDF3;font-size:14px;line-height:1.8;margin:0'>
        This project develops an end-to-end brain tumor MRI classification system using
        the <b style='color:#58A6FF'>Vision Transformer (ViT-B16)</b> architecture pre-trained on
        ImageNet-21k. The system classifies MRI scans into four categories —
        Glioma, Meningioma, Pituitary Tumor, and No Tumor — with a test accuracy of
        <b style='color:#3FB950'>97.23%</b> and a Macro F1 score of <b style='color:#3FB950'>0.97</b>.
        <br><br>
        To address the critical need for explainability in medical AI,
        <b style='color:#FF6B6B'>Grad-CAM</b> (Gradient-weighted Class Activation Mapping)
        is integrated to generate attention heatmaps showing which brain regions
        the model focused on when making each prediction.
        The complete system is deployed as a
        <b style='color:#FFB347'>live web application on Hugging Face Spaces</b>,
        accessible to medical professionals and researchers worldwide.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Architecture ───────────────────────────────────────────────────────
st.markdown("### 🏗️ Model Architecture")

steps = [
    ("Input",        "224 × 224 × 3 MRI image",         "#1F6FEB"),
    ("Patch Embed",  "196 patches of 16×16px → 768-dim", "#388BFD"),
    ("Transformer",  "12 blocks: MSA + MLP + LayerNorm", "#58A6FF"),
    ("CLS Token",    "768-dimensional feature vector",   "#79C0FF"),
    ("Head Layer 1", "Linear(768→256) + GELU + Drop 0.3","#D2A8FF"),
    ("Head Layer 2", "Linear(256→128) + GELU + Drop 0.3","#BC8CFF"),
    ("Output",       "Linear(128→4) → Softmax",          "#3FB950"),
]

cols = st.columns(len(steps))
for i, (col, (name, detail, color)) in enumerate(zip(cols, steps)):
    with col:
        arrow = "→" if i < len(steps)-1 else ""
        st.markdown(f"""
        <div style='background:{color}20;border:1px solid {color}60;border-radius:8px;
                    padding:10px 8px;text-align:center;font-size:11px'>
            <div style='font-weight:700;color:{color};margin-bottom:4px'>{name}</div>
            <div style='color:#8B949E;line-height:1.4'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Training ───────────────────────────────────────────────────────────
st.markdown("### 📚 Two-Phase Training Strategy")
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div style='background:#161B22;border:1px solid #4F8BF9;border-radius:12px;padding:16px 20px'>
        <div style='font-size:15px;font-weight:700;color:#4F8BF9;margin-bottom:10px'>
            Phase 1 — Head Only (5 Epochs)
        </div>
        <div style='font-size:13px;color:#8B949E;line-height:1.8'>
            ✅ Backbone: <b style='color:#E6EDF3'>Frozen</b><br>
            ✅ Optimizer: <b style='color:#E6EDF3'>Adam, LR=1e-3</b><br>
            ✅ Trainable params: <b style='color:#E6EDF3'>230,276</b><br>
            ✅ Best Val Accuracy: <b style='color:#3FB950'>85.52%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style='background:#161B22;border:1px solid #3FB950;border-radius:12px;padding:16px 20px'>
        <div style='font-size:15px;font-weight:700;color:#3FB950;margin-bottom:10px'>
            Phase 2 — Full Fine-Tuning (up to 45 Epochs)
        </div>
        <div style='font-size:13px;color:#8B949E;line-height:1.8'>
            ✅ Backbone: <b style='color:#E6EDF3'>Unfrozen</b><br>
            ✅ Optimizer: <b style='color:#E6EDF3'>AdamW, LR=1e-4</b><br>
            ✅ AMP + GradClip + EarlyStopping<br>
            ✅ Best Val Accuracy: <b style='color:#3FB950'>97.23%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Tech stack ─────────────────────────────────────────────────────────
st.markdown("### ⚙️ Technology Stack")

tech_categories = {
    "🧠 Deep Learning": [
        ("PyTorch 2.x",    "Primary training framework"),
        ("Timm",           "Pre-trained ViT-B16 (ImageNet-21k)"),
        ("TensorFlow 2.x", "Alternative model support"),
        ("TorchVision",    "Image transforms & DataLoader"),
    ],
    "📦 Data Processing": [
        ("Albumentations", "9-technique MRI augmentation pipeline"),
        ("Pillow",         "Image loading & preprocessing"),
        ("OpenCV",         "Grad-CAM heatmap overlay generation"),
        ("scikit-learn",   "Metrics, stratified split"),
    ],
    "🚀 Deployment": [
        ("Streamlit",       "This interactive web dashboard"),
        ("Gradio",          "Live classifier on Hugging Face"),
        ("Hugging Face",    "Spaces (app) + Hub (model weights)"),
        ("Git LFS",         "Large model file storage"),
    ],
    "📊 Visualization": [
        ("Plotly",      "Interactive charts & confusion matrix"),
        ("Matplotlib",  "Training curves & static plots"),
        ("Seaborn",     "Heatmap styling"),
        ("Streamlit",   "Embedded dashboard components"),
    ],
}

for category, tools in tech_categories.items():
    st.markdown(f"#### {category}")
    cols = st.columns(4)
    for (name, desc), col in zip(tools, cols):
        with col:
            st.markdown(f"""
            <div style='background:#161B22;border:1px solid #30363D;border-radius:8px;
                        padding:10px;margin-bottom:10px'>
                <div style='font-size:13px;font-weight:600;color:#E6EDF3'>{name}</div>
                <div style='font-size:11px;color:#8B949E;margin-top:3px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Deployment links ───────────────────────────────────────────────────
st.markdown("### 🔗 Deployment & Resources")
d1, d2, d3 = st.columns(3)

with d1:
    st.markdown("""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:12px;
                padding:16px;text-align:center'>
        <div style='font-size:24px'>🤗</div>
        <div style='font-size:14px;font-weight:600;color:#E6EDF3;margin-top:8px'>Live App</div>
        <div style='font-size:12px;color:#8B949E;margin-top:4px;margin-bottom:12px'>
            Hugging Face Spaces<br>24/7 · Free · No login
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.link_button("Open Live Classifier →",
                   "https://huggingface.co/spaces/steve2514/brain-tumor-mri-classifier",
                   use_container_width=True)

with d2:
    st.markdown("""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:12px;
                padding:16px;text-align:center'>
        <div style='font-size:24px'>📦</div>
        <div style='font-size:14px;font-weight:600;color:#E6EDF3;margin-top:8px'>Model Weights</div>
        <div style='font-size:12px;color:#8B949E;margin-top:4px;margin-bottom:12px'>
            Hugging Face Hub<br>984MB PyTorch checkpoint
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.link_button("View Model on HF Hub →",
                   "https://huggingface.co/steve2514/brain-tumor-project",
                   use_container_width=True)

with d3:
    st.markdown("""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:12px;
                padding:16px;text-align:center'>
        <div style='font-size:24px'>📂</div>
        <div style='font-size:14px;font-weight:600;color:#E6EDF3;margin-top:8px'>Dataset</div>
        <div style='font-size:12px;color:#8B949E;margin-top:4px;margin-bottom:12px'>
            Kaggle · 7,200 MRI images<br>4 balanced classes
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.link_button("View Dataset on Kaggle →",
                   "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
                   use_container_width=True)

st.markdown("---")
st.markdown("""
<div class='disclaimer'>
    <b style='color:#FFB347'>⚠️ Medical Disclaimer:</b>
    This application is for research and educational purposes only.
    It is NOT approved for clinical diagnosis and must NOT be used as a substitute
    for professional medical advice or treatment by a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
