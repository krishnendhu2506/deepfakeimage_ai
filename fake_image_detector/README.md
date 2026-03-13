# AI Image Authenticity Detector using Deep Learning

This project is a Flask web application that classifies uploaded images as `Real Image` or `AI Generated Image` using a PyTorch CNN. It also generates Grad-CAM visual explanations, prediction confidence scores, dashboard analytics, and validation artifacts.

## Project Structure

```text
fake_image_detector/
├── app.py
├── model/
│   ├── fake_image_detector.pth
│   ├── network.py
│   ├── predict.py
│   └── train_model.py
├── static/
│   ├── css/style.css
│   ├── generated/
│   ├── images/
│   ├── js/app.js
│   └── uploads/
├── templates/
│   ├── dashboard.html
│   ├── index.html
│   ├── result.html
│   └── upload.html
├── utils/
│   ├── gradcam_visualization.py
│   └── image_preprocess.py
├── prediction_history.json
├── README.md
└── requirements.txt
```

## Installation

```powershell
pip install -r requirements.txt
```

or:

```powershell
pip install torch torchvision flask opencv-python numpy pandas matplotlib scikit-learn pillow
```

## Train

```powershell
cd C:\Users\ASUS\OneDrive\Desktop\deepfakeimage_ai\fake_image_detector
python model\train_model.py
```

The trained model is saved as:

```text
model\fake_image_detector.pth
```

## Run

```powershell
cd C:\Users\ASUS\OneDrive\Desktop\deepfakeimage_ai\fake_image_detector
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Notes

- The training script scans both `dataset/` and `Dataset/` layouts automatically.
- Validation metrics are written to `static/generated/training_metrics.json`.
- Confusion matrix and loss graph are saved in `static/generated/`.
