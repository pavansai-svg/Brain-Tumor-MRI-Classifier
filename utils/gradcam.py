"""
utils/gradcam.py
Grad-CAM for ViT-B16 (PyTorch) and standard CNN (TensorFlow).
"""

import numpy as np
import cv2
from PIL import Image


# ══════════════════════════════════════════════════════════
#  PYTORCH GRAD-CAM (ViT-B16)
# ══════════════════════════════════════════════════════════

class ViTGradCAM:
    """
    Grad-CAM for Vision Transformer (ViT-B16).
    Hooks into the last transformer block's normalization layer.
    """

    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook last transformer block norm1
        target_layer = model.backbone.blocks[-1].norm1
        self._hook_f = target_layer.register_forward_hook(self._save_activation)
        self._hook_b = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            img_tensor: (1, 3, 224, 224) torch tensor on device
            class_idx:  target class (None = predicted class)

        Returns:
            heatmap: (224, 224) float32 array normalized 0-1
        """
        import torch

        self.model.zero_grad()
        img_tensor = img_tensor.clone().requires_grad_(True)

        with torch.enable_grad():
            output   = self.model(img_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[0, class_idx]
            score.backward()

        # grads & activations: (num_tokens, 768)
        grads = self.gradients[0]    # (197, 768)
        acts  = self.activations[0]  # (197, 768)

        # Weight activations by mean gradient
        weights = grads.mean(dim=-1)              # (197,)
        cam     = (weights.unsqueeze(-1) * acts).sum(dim=-1)  # (197,)

        # Remove CLS token → reshape to 14×14 patch grid
        cam = cam[1:]                             # (196,)
        cam = cam.reshape(14, 14).cpu().numpy()
        cam = np.maximum(cam, 0)                  # ReLU
        if cam.max() > 0:
            cam = cam / cam.max()                 # Normalize 0-1

        # Upsample to 224×224
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cam

    def remove_hooks(self):
        self._hook_f.remove()
        self._hook_b.remove()


# ══════════════════════════════════════════════════════════
#  TENSORFLOW GRAD-CAM (last conv layer)
# ══════════════════════════════════════════════════════════

def compute_tf_gradcam(model, img_array: np.ndarray, class_idx: int | None = None) -> np.ndarray:
    """
    Grad-CAM for TensorFlow model.
    Automatically finds the last Conv2D layer.

    Args:
        model:     tf.keras.Model
        img_array: (1, 224, 224, 3) normalized float32 array
        class_idx: target class (None = predicted class)

    Returns:
        heatmap: (224, 224) float32 array normalized 0-1
    """
    try:
        import tensorflow as tf

        # Find last Conv2D layer
        conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer = layer
                break

        if conv_layer is None:
            # Fallback: return blank heatmap
            return np.zeros((224, 224), dtype=np.float32)

        grad_model = tf.keras.Model(
            inputs  = model.inputs,
            outputs = [conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0]).numpy()
            score = predictions[:, class_idx]

        grads     = tape.gradient(score, conv_outputs)[0]  # (H, W, C)
        weights   = tf.reduce_mean(grads, axis=(0, 1))     # (C,)
        cam       = tf.reduce_sum(weights * conv_outputs[0], axis=-1).numpy()
        cam       = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cam

    except Exception:
        return np.zeros((224, 224), dtype=np.float32)


# ══════════════════════════════════════════════════════════
#  OVERLAY HEATMAP ON IMAGE
# ══════════════════════════════════════════════════════════

def overlay_heatmap(
    pil_image : Image.Image,
    heatmap   : np.ndarray,
    alpha     : float = 0.45,
    colormap  : int   = cv2.COLORMAP_JET
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap onto an MRI image.

    Args:
        pil_image: original PIL image
        heatmap:   (224, 224) float32 array, values 0-1
        alpha:     heatmap opacity (0=invisible, 1=full heatmap)
        colormap:  OpenCV colormap constant

    Returns:
        overlay PIL image (RGB)
    """
    # Resize original to 224×224
    orig = np.array(pil_image.convert("RGB").resize((224, 224)))

    # Convert heatmap to color
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(orig, 1 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(overlay)


def get_attention_regions(heatmap: np.ndarray, threshold: float = 0.6) -> str:
    """Describe which quadrant of the image has the highest attention."""
    h, w  = heatmap.shape
    quad  = {
        "Top-Left"    : heatmap[:h//2,  :w//2 ].mean(),
        "Top-Right"   : heatmap[:h//2,   w//2:].mean(),
        "Bottom-Left" : heatmap[ h//2:, :w//2 ].mean(),
        "Bottom-Right": heatmap[ h//2:,  w//2:].mean(),
    }
    top_region = max(quad, key=quad.get)
    top_val    = quad[top_region]

    if top_val < threshold * heatmap.max():
        return "Diffuse — attention spread across the entire image"
    return f"Concentrated in the {top_region} region of the MRI"
