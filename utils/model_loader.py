"""
utils/model_loader.py
Loads PyTorch ViT-B16 and TensorFlow models with Streamlit caching.
"""

import streamlit as st
import os
import numpy as np
from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────
CLASS_NAMES  = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
IMG_SIZE     = 224

RISK_CONFIG = {
    "glioma_tumor"    : ("HIGH",   "#FF6B6B", "#FF000018"),
    "meningioma_tumor": ("MEDIUM", "#FFB347", "#FFA50018"),
    "no_tumor"        : ("LOW",    "#4CAF50", "#00FF0018"),
    "pituitary_tumor" : ("MEDIUM", "#FFB347", "#FFA50018"),
}


# ══════════════════════════════════════════════════════════
#  PYTORCH MODEL
# ══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading PyTorch model...")
def load_pytorch_model(model_path: str):
    """Load ViT-B16 PyTorch model. Returns (model, device) or (None, None) on error."""
    try:
        import torch
        import torch.nn as nn
        import timm

        class BrainTumorViT(nn.Module):
            def __init__(self, num_classes=4, dropout=0.3):
                super().__init__()
                self.backbone = timm.create_model(
                    "vit_base_patch16_224",
                    pretrained=False,
                    num_classes=0
                )
                self.head = nn.Sequential(
                    nn.Linear(768, 256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(128, num_classes)
                )
            def forward(self, x):
                return self.head(self.backbone(x))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = BrainTumorViT(num_classes=4).to(device)

        ckpt = torch.load(model_path, map_location=device)

        # Handle both plain state_dict and full checkpoint
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "model_state" in ckpt:
                state_dict = ckpt["model_state"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.eval()
        return model, device

    except Exception as e:
        st.error(f"❌ Failed to load PyTorch model: {e}")
        return None, None


def predict_pytorch(model, device, pil_image: Image.Image) -> dict:
    """Run inference with PyTorch model. Returns prediction dict."""
    try:
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

        img_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx  = int(np.argmax(probs))
        pred_cls  = CLASS_NAMES[pred_idx]
        risk, risk_color, risk_bg = RISK_CONFIG[pred_cls]

        return {
            "class_name"   : pred_cls,
            "class_label"  : CLASS_LABELS[pred_idx],
            "confidence"   : float(probs[pred_idx]) * 100,
            "probabilities": {CLASS_LABELS[i]: float(probs[i]) for i in range(4)},
            "risk"         : risk,
            "risk_color"   : risk_color,
            "risk_bg"      : risk_bg,
            "pred_idx"     : pred_idx,
            "probs_array"  : probs,
            "img_tensor"   : img_tensor,
        }

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return {}


# ══════════════════════════════════════════════════════════
#  TENSORFLOW MODEL
# ══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading TensorFlow model...")
def load_tensorflow_model(model_path: str):
    """Load TensorFlow .keras model. Returns model or None on error."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load TensorFlow model: {e}")
        return None


def predict_tensorflow(model, pil_image: Image.Image) -> dict:
    """Run inference with TensorFlow model. Returns prediction dict."""
    try:
        import tensorflow as tf
        import numpy as np

        img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        arr = np.expand_dims(arr, axis=0)

        probs = model.predict(arr, verbose=0)[0]

        pred_idx  = int(np.argmax(probs))
        pred_cls  = CLASS_NAMES[pred_idx]
        risk, risk_color, risk_bg = RISK_CONFIG[pred_cls]

        return {
            "class_name"   : pred_cls,
            "class_label"  : CLASS_LABELS[pred_idx],
            "confidence"   : float(probs[pred_idx]) * 100,
            "probabilities": {CLASS_LABELS[i]: float(probs[i]) for i in range(4)},
            "risk"         : risk,
            "risk_color"   : risk_color,
            "risk_bg"      : risk_bg,
            "pred_idx"     : pred_idx,
            "probs_array"  : probs,
        }

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return {}


# ══════════════════════════════════════════════════════════
#  MODEL FILE FINDER
# ══════════════════════════════════════════════════════════

def find_model_path(filename: str) -> str | None:
    """Search common locations for model files."""
    search_dirs = [
        "models",
        ".",
        os.path.expanduser("~"),
        "/kaggle/input",
    ]
    for d in search_dirs:
        full = os.path.join(d, filename)
        if os.path.exists(full):
            return full
    return None
