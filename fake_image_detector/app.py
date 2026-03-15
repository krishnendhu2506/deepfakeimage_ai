import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from utils.image_preprocess import allowed_file, ensure_directories


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
GENERATED_DIR = BASE_DIR / "static" / "generated"
HISTORY_FILE = BASE_DIR / "prediction_history.json"
MODEL_PATH = BASE_DIR / "model" / "fake_image_detector.pth"

app = Flask(__name__)
app.secret_key = "fake-image-detector-secret-key"

ensure_directories([UPLOAD_DIR, GENERATED_DIR])

MODEL_CACHE = None
MODEL_ERROR = None


def read_json_file(file_path: Path, default):
    if not file_path.exists():
        return default
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default


def write_json_file(file_path: Path, payload):
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def get_model():
    global MODEL_CACHE, MODEL_ERROR
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    if not MODEL_PATH.exists():
        MODEL_ERROR = "Model file not found. Train the model first using model/train_model.py."
        return None

    try:
        from model.predict import load_detector

        MODEL_CACHE = load_detector(MODEL_PATH)
        MODEL_ERROR = None
        return MODEL_CACHE
    except Exception as error:
        MODEL_ERROR = f"Model loading failed: {error}"
        return None


def load_history():
    return read_json_file(HISTORY_FILE, [])


def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    write_json_file(HISTORY_FILE, history[:100])


def get_training_metrics():
    metrics_file = GENERATED_DIR / "training_metrics.json"
    metrics = read_json_file(metrics_file, {})
    if not metrics:
        return {
            "accuracy": None,
            "loss": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "notes": "Run model/train_model.py to generate validation metrics.",
        }
    return metrics


def build_dashboard_data():
    history = load_history()
    real_count = sum(1 for item in history if item["label"] == "Real Image")
    fake_count = sum(1 for item in history if item["label"] == "AI Generated Image")
    return {
        "total_predictions": len(history),
        "real_count": real_count,
        "fake_count": fake_count,
        "history": history[:12],
        "training_metrics": get_training_metrics(),
        "model_error": MODEL_ERROR,
        "model_available": MODEL_PATH.exists(),
    }


@app.route("/")
def index():
    dashboard = build_dashboard_data()
    return render_template("index.html", dashboard=dashboard)


@app.route("/upload")
def upload():
    get_model()
    model_available = MODEL_PATH.exists() and MODEL_ERROR is None
    return render_template("upload.html", model_available=model_available, model_error=MODEL_ERROR)


@app.route("/predict", methods=["POST"])
def predict():
    detector = get_model()
    if detector is None:
        flash(MODEL_ERROR or "Model could not be loaded.", "error")
        return redirect(url_for("upload"))

    if "image" not in request.files:
        flash("No image file was uploaded.", "error")
        return redirect(url_for("upload"))

    file = request.files["image"]
    if file.filename == "":
        flash("Please choose an image file.", "error")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash("Only JPG, JPEG, and PNG files are allowed.", "error")
        return redirect(url_for("upload"))

    analysis_id = uuid.uuid4().hex
    extension = Path(file.filename).suffix.lower()
    upload_name = f"{analysis_id}{extension}"
    upload_path = UPLOAD_DIR / upload_name
    file.save(upload_path)

    try:
        from model.predict import predict_image

        result = predict_image(
            image_path=upload_path,
            model_bundle=detector,
            output_dir=GENERATED_DIR,
            analysis_id=analysis_id,
        )
    except Exception as error:
        flash(f"Prediction failed: {error}", "error")
        return redirect(url_for("upload"))

    result_payload = {
        "id": analysis_id,
        "filename": upload_name,
        "label": result["label"],
        "confidence": result["confidence"],
        "explanation": result["explanation"],
        "probabilities": result["probabilities"],
        "original_image": result["original_image"],
        "heatmap_image": result["heatmap_image"],
        "overlay_image": result["overlay_image"],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_history_entry(result_payload)
    result_file = GENERATED_DIR / f"{analysis_id}.json"
    write_json_file(result_file, result_payload)
    return redirect(url_for("result", analysis_id=analysis_id))


@app.route("/result/<analysis_id>")
def result(analysis_id):
    result_file = GENERATED_DIR / f"{analysis_id}.json"
    result_payload = read_json_file(result_file, None)
    if result_payload is None:
        flash("Prediction result not found.", "error")
        return redirect(url_for("upload"))
    return render_template("result.html", result=result_payload)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", dashboard=build_dashboard_data())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
