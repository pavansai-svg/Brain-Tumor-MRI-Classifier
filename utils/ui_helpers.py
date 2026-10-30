"""
utils/ui_helpers.py
Reusable Streamlit UI components shared across pages.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
CLASS_COLORS = ["#FF6B6B", "#FFB347", "#4CAF50", "#58A6FF"]


def inject_page_css():
    """Inject shared CSS for all pages (called at top of each page)."""
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
        border-right: 1px solid #30363D;
    }
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    [data-testid="metric-container"] {
        background: #161B22; border: 1px solid #30363D;
        border-radius: 12px; padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { font-size: 24px !important; color: #E6EDF3 !important; }
    [data-testid="stMetricLabel"] { font-size: 12px !important; color: #8B949E !important; }
    .section-title {
        font-size: 22px; font-weight: 700; color: #E6EDF3;
        margin-bottom: 4px; padding-bottom: 8px;
        border-bottom: 2px solid #4F8BF9;
    }
    .section-sub { font-size: 13px; color: #8B949E; margin-bottom: 20px; }
    .disclaimer {
        background: #1C2128; border-left: 3px solid #FFB347;
        border-radius: 0 8px 8px 0; padding: 10px 16px;
        font-size: 12px; color: #8B949E; margin-top: 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1F6FEB, #4F8BF9);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar_nav():
    """Render consistent sidebar navigation on every page."""
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 12px 0 16px;'>
            <div style='font-size:32px'>🧠</div>
            <div style='font-size:15px; font-weight:700; color:#E6EDF3; margin-top:6px'>
                Brain Tumor MRI
            </div>
            <div style='font-size:11px; color:#8B949E; margin-top:3px'>
                ViT-B16 · 97.23% Accuracy
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.page_link("app.py",                label="🏠  Home")
        st.page_link("pages/1_Prediction.py", label="🔬  Prediction")
        st.page_link("pages/2_GradCAM.py",    label="🔥  Grad-CAM")
        st.page_link("pages/3_Dashboard.py",  label="📊  Dashboard")
        st.page_link("pages/4_About.py",      label="ℹ️   About")

        st.markdown("---")
        st.markdown("""
        <div style='font-size:11px;color:#8B949E;text-align:center;padding:4px'>
            <a href='https://huggingface.co/spaces/steve2514/brain-tumor-mri-classifier'
               style='color:#58A6FF;text-decoration:none'>🤗 Live Demo</a>
        </div>
        """, unsafe_allow_html=True)


def model_selector() -> tuple[str, str]:
    """
    Sidebar model selection widget.
    Returns (model_type, model_path).
    model_type is 'pytorch' or 'tensorflow'.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size:11px;color:#8B949E;padding:0 8px;margin-bottom:8px'>
        MODEL SELECTION
    </div>
    """, unsafe_allow_html=True)

    model_type = st.sidebar.radio(
        "Select model",
        ["PyTorch (ViT-B16)", "TensorFlow (.keras)"],
        label_visibility="collapsed"
    )

    if model_type == "PyTorch (ViT-B16)":
        default_path = "models/best_model_pytorch.pth"
        model_path   = st.sidebar.text_input("PyTorch model path", value=default_path)
        return "pytorch", model_path
    else:
        default_path = "models/best_model_phase1.keras"
        model_path   = st.sidebar.text_input("TensorFlow model path", value=default_path)
        return "tensorflow", model_path


def render_prediction_result(result: dict):
    """Render a styled prediction result box."""
    if not result:
        return

    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(result["risk"], "⚪")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1C2942,#162032);
                border:1px solid #4F8BF9;border-radius:16px;
                padding:24px;text-align:center;margin-bottom:16px'>
        <div style='font-size:13px;color:#8B949E;margin-bottom:6px'>Predicted Class</div>
        <div style='font-size:30px;font-weight:700;color:#58A6FF;margin-bottom:6px'>
            {result["class_label"]}
        </div>
        <div style='font-size:16px;color:#E6EDF3;margin-bottom:12px'>
            Confidence: <b style='color:#58A6FF'>{result["confidence"]:.1f}%</b>
        </div>
        <div style='display:inline-block;padding:6px 18px;border-radius:20px;
                    background:{result["risk_bg"]};border:1px solid {result["risk_color"]}50;
                    font-size:14px;font-weight:700;color:{result["risk_color"]}'>
            {risk_emoji} Risk Level: {result["risk"]}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_chart(probabilities: dict) -> go.Figure:
    """Render interactive Plotly probability bar chart."""
    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(
            color=CLASS_COLORS,
            line=dict(color="#30363D", width=0.5)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#E6EDF3", size=12)
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 115], showgrid=True,
                   gridcolor="#30363D", title="Probability (%)",
                   tickfont=dict(color="#8B949E")),
        yaxis=dict(tickfont=dict(color="#E6EDF3")),
        margin=dict(t=10, b=10, l=10, r=60),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B949E"),
        showlegend=False,
    )
    return fig


def render_confidence_gauge(confidence: float) -> go.Figure:
    """Render a confidence gauge chart."""
    color = "#4CAF50" if confidence >= 90 else "#FFB347" if confidence >= 70 else "#FF6B6B"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = confidence,
        title = dict(text="Confidence %", font=dict(color="#8B949E", size=13)),
        number= dict(suffix="%", font=dict(color="#E6EDF3", size=24)),
        gauge = dict(
            axis    = dict(range=[0, 100], tickcolor="#8B949E",
                           tickfont=dict(color="#8B949E")),
            bar     = dict(color=color),
            bgcolor = "#161B22",
            bordercolor="#30363D",
            steps=[
                dict(range=[0,  50], color="#1C1C2E"),
                dict(range=[50, 70], color="#1C2128"),
                dict(range=[70,100], color="#1C2128"),
            ]
        )
    ))
    fig.update_layout(
        height=200, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B949E")
    )
    return fig


def show_image_info(image_info: dict):
    """Display image metadata."""
    st.markdown(f"""
    <div style='background:#161B22;border:1px solid #30363D;border-radius:8px;
                padding:10px 14px;font-size:12px;color:#8B949E'>
        <span style='color:#E6EDF3'>Size:</span> {image_info['width']}×{image_info['height']}px &nbsp;
        <span style='color:#E6EDF3'>Mode:</span> {image_info['mode']} &nbsp;
        <span style='color:#E6EDF3'>Format:</span> {image_info.get('format','N/A')}
    </div>
    """, unsafe_allow_html=True)


def disclaimer_box():
    """Medical disclaimer shown on every prediction page."""
    st.markdown("""
    <div class='disclaimer'>
        <b style='color:#FFB347'>⚠️ Medical Disclaimer:</b>
        This tool is for research and educational purposes only.
        It is NOT approved for clinical use. Always consult a licensed physician.
    </div>
    """, unsafe_allow_html=True)
