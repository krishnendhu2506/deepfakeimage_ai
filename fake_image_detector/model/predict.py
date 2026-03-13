from pathlib import Path

import torch

from model.network import FakeImageDetectorCNN


CLASS_NAMES = {0: "Real Image", 1: "AI Generated Image"}


def load_detector(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FakeImageDetectorCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {"model": model, "device": device}


def build_explanation(predicted_class, confidence_score):
    if predicted_class == 1:
        if confidence_score >= 0.85:
            return "Detected visual artifacts, repetitive textures, or synthetic patterns commonly associated with AI generated imagery."
        return "The image shows several cues that lean toward AI synthesis, including texture smoothness and localized inconsistencies."
    if confidence_score >= 0.85:
        return "The image contains natural lighting, realistic texture transitions, and spatial patterns consistent with real photography."
    return "The detector found more natural photographic structure than synthetic artifacts, but the decision is moderately confident."


def predict_image(image_path, model_bundle, output_dir, analysis_id):
    from utils.gradcam_visualization import generate_gradcam_visuals
    from utils.image_preprocess import preprocess_single_image

    model = model_bundle["model"]
    device = model_bundle["device"]
    image_tensor = preprocess_single_image(image_path, as_tensor=True).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits).item()

    predicted_class = 1 if probability >= 0.5 else 0
    confidence = probability if predicted_class == 1 else 1 - probability

    probabilities = {
        "Real Image": round((1 - probability) * 100, 2),
        "AI Generated Image": round(probability * 100, 2),
    }

    gradcam_output = generate_gradcam_visuals(
        model=model,
        device=device,
        image_path=Path(image_path),
        output_dir=Path(output_dir),
        analysis_id=analysis_id,
    )

    return {
        "label": CLASS_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities,
        "explanation": build_explanation(predicted_class, confidence),
        "original_image": gradcam_output["original_image"],
        "heatmap_image": gradcam_output["heatmap_image"],
        "overlay_image": gradcam_output["overlay_image"],
    }
