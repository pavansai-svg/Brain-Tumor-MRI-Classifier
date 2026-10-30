"""
utils/preprocessing.py
Image loading, validation, and preprocessing helpers.
"""

import numpy as np
from PIL import Image
import io
import streamlit as st

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image_from_upload(uploaded_file) -> Image.Image | None:
    """
    Load a PIL image from a Streamlit UploadedFile.
    Returns None if the file is invalid.
    """
    if uploaded_file is None:
        return None
    try:
        raw = uploaded_file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        st.error(f"❌ Could not open image: {e}")
        return None


def validate_image(pil_image: Image.Image) -> bool:
    """Basic sanity check on the uploaded image."""
    if pil_image is None:
        return False
    w, h = pil_image.size
    if w < 32 or h < 32:
        st.warning("⚠️ Image resolution is very low. Results may be inaccurate.")
    if pil_image.mode not in ("RGB", "L", "RGBA"):
        st.warning(f"⚠️ Unusual image mode: {pil_image.mode}. Converting to RGB.")
    return True


def preprocess_for_display(pil_image: Image.Image, target_size: int = IMG_SIZE) -> Image.Image:
    """Resize image for display — keeps aspect ratio with padding."""
    return pil_image.convert("RGB").resize((target_size, target_size), Image.LANCZOS)


def pil_to_numpy_normalized(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to ImageNet-normalized float32 array (H, W, 3)."""
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return arr


def numpy_to_display(arr: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization for display (clipped to 0-255 uint8)."""
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def get_image_info(pil_image: Image.Image) -> dict:
    """Return metadata about an uploaded image."""
    return {
        "width"  : pil_image.width,
        "height" : pil_image.height,
        "mode"   : pil_image.mode,
        "format" : getattr(pil_image, "format", "Unknown"),
    }
