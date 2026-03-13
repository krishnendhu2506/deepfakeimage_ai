from pathlib import Path

import cv2
import numpy as np
import torch

from utils.image_preprocess import preprocess_single_image


def make_gradcam_heatmap(model, image_tensor, device):
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.features[6]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(image_tensor.to(device))
    score = torch.sigmoid(logits)[0, 0]
    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    activation = activations[0][0]
    gradient = gradients[0][0]
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = (weights * activation).sum(dim=0)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam.cpu().numpy()


def save_visuals(original_bgr, heatmap, output_dir, analysis_id):
    height, width = original_bgr.shape[:2]
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (width, height))
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.6, colored_heatmap, 0.4, 0)

    original_path = output_dir / f"{analysis_id}_original.jpg"
    heatmap_path = output_dir / f"{analysis_id}_heatmap.jpg"
    overlay_path = output_dir / f"{analysis_id}_overlay.jpg"

    cv2.imwrite(str(original_path), original_bgr)
    cv2.imwrite(str(heatmap_path), colored_heatmap)
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "original_image": f"generated/{original_path.name}",
        "heatmap_image": f"generated/{heatmap_path.name}",
        "overlay_image": f"generated/{overlay_path.name}",
    }


def generate_gradcam_visuals(model, device, image_path, output_dir, analysis_id):
    image_tensor = preprocess_single_image(image_path, as_tensor=True)
    heatmap = make_gradcam_heatmap(model, image_tensor, device)

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise ValueError(f"Could not load image for Grad-CAM: {image_path}")

    return save_visuals(original_bgr, heatmap, output_dir, analysis_id)
