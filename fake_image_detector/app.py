import json
import os
<<<<<<< HEAD
import tempfile
=======
>>>>>>> 5cc97653d18015ff80dfb1f637839866e4bbac15
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for

from utils.image_preprocess import allowed_file, ensure_directories


BASE_DIR = Path(__file__).resolve().parent
STATIC_GENERATED_DIR = BASE_DIR / "static" / "generated"
RUNTIME_BASE_DIR = Path(os.getenv("FAKE_IMAGE_RUNTIME_DIR", Path(tempfile.gettempdir()) / "fake_image_detector"))
UPLOAD_DIR = RUNTIME_BASE_DIR / "uploads"
GENERATED_DIR = RUNTIME_BASE_DIR / "generated"
HISTORY_FILE = RUNTIME_BASE_DIR / "prediction_history.json"
MODEL_PATH = BASE_DIR / "model" / "fake_image_detector.pth"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fake-image-detector-secret-key")

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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def asset_url(asset_path: str | None):
    if not asset_path:
        return ""
    if asset_path.startswith("generated/") or asset_path.startswith("uploads/"):
        return url_for("media_file", file_path=asset_path)
    return url_for("static", filename=asset_path)


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
    metrics = read_json_file(STATIC_GENERATED_DIR / "training_metrics.json", {})
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


@app.context_processor
def inject_asset_helpers():
    return {"asset_url": asset_url}


@app.route("/")
def index():
    dashboard = build_dashboard_data()
    return render_template("index.html", dashboard=dashboard)


@app.route("/upload")
def upload():
    get_model()
    model_available = MODEL_PATH.exists() and MODEL_ERROR is None
    return render_template("upload.html", model_available=model_available, model_error=model_error_text())


@app.route("/predict", methods=["POST"])
def predict():
    detector = get_model()
    if detector is None:
        flash(model_error_text() or "Model could not be loaded.", "error")
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
    write_json_file(GENERATED_DIR / f"{analysis_id}.json", result_payload)
    return render_template("result.html", result=result_payload)


@app.route("/result/<analysis_id>")
def result(analysis_id):
    result_payload = read_json_file(GENERATED_DIR / f"{analysis_id}.json", None)
    if result_payload is None:
        flash("Prediction result not found.", "error")
        return redirect(url_for("upload"))
    return render_template("result.html", result=result_payload)


@app.route("/media/<path:file_path>")
def media_file(file_path):
    normalized = Path(file_path)
    if len(normalized.parts) != 2 or normalized.parts[0] not in {"generated", "uploads"}:
        abort(404)

    directory = GENERATED_DIR if normalized.parts[0] == "generated" else UPLOAD_DIR
    return send_from_directory(directory, normalized.parts[1])


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", dashboard=build_dashboard_data())


def model_error_text():
    return MODEL_ERROR


if __name__ == "__main__":
<<<<<<< HEAD
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
=======
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
>>>>>>> 5cc97653d18015ff80dfb1f637839866e4bbac15
