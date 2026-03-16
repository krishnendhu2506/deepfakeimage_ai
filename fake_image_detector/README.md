# AI Image Authenticity Detector using Deep Learning

AI Image Authenticity Detector is a Flask + PyTorch web application that classifies uploaded images as `Real Image` or `AI Generated Image`. The app combines binary classification, confidence scoring, Grad-CAM visualization, and dashboard analytics in one responsive interface.

## Core Features

- Flask backend for upload, inference, results, and dashboard routes
- PyTorch CNN model for real-vs-fake image classification
- Grad-CAM heatmap generation with original, heatmap, and overlay views
- Confidence score and probability bar chart
- Prediction explanation text for user-friendly interpretation
- Prediction history dashboard with validation metrics
- Responsive frontend with TailwindCSS styling, custom CSS, and light GSAP animations

## Tech Stack

- Backend: Flask
- Deep Learning: PyTorch
- Image Processing: OpenCV
- Frontend: HTML, CSS, JavaScript
- Charts: Chart.js
- Animations: GSAP

## Project Structure

```text
fake_image_detector/
|-- app.py
|-- wsgi.py
|-- Procfile
|-- requirements.txt
|-- README.md
|-- DEMO_SCRIPT.md
|-- prediction_history.json
|-- model/
|   |-- fake_image_detector.pth
|   |-- network.py
|   |-- predict.py
|   `-- train_model.py
|-- static/
|   |-- css/style.css
|   |-- generated/
|   |-- js/app.js
|   `-- uploads/
|-- templates/
|   |-- index.html
|   |-- upload.html
|   |-- result.html
|   `-- dashboard.html
`-- utils/
    |-- image_preprocess.py
    `-- gradcam_visualization.py
```

## Local Setup

### 1. Open the project

```powershell
cd C:\Users\ASUS\OneDrive\Desktop\deepfakeimage_ai\fake_image_detector
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Training the Model

```powershell
python model\train_model.py
```

This will:

- scan the available dataset folders
- create a train/validation split automatically
- train the PyTorch CNN
- save the model to `model/fake_image_detector.pth`
- write validation metrics to `static/generated/training_metrics.json`
- save the confusion matrix and loss graph in `static/generated/`

## Running the App Locally

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Railway Deployment

This repository is now prepared for Railway using a root-level [`railway.json`](/c:/Users/ASUS/OneDrive/Desktop/deepfakeimage_ai/railway.json) and [`Procfile`](/c:/Users/ASUS/OneDrive/Desktop/deepfakeimage_ai/Procfile).

### What was added for Railway

- root-level `railway.json` so Railway can build from the repo root even though the Flask app lives in `fake_image_detector/`
- root-level `Procfile` with the production start command
- `gunicorn` in `requirements.txt`
- host/port aware Flask runtime in `app.py`

### Railway Deploy Steps

1. Push the repository to GitHub.
2. Sign in to Railway.
3. Click `New Project`.
4. Click `Deploy from GitHub Repo`.
5. Select your repository.
6. Railway should detect the repo and use `railway.json` automatically.
7. Open the created service.
8. Go to the `Variables` tab.
9. Add:
   - `FLASK_SECRET_KEY` = a long random string
10. Optional but recommended for persistent uploads/history:
   - create a Railway Volume
   - mount it to `/data`
   - set `FAKE_IMAGE_RUNTIME_DIR=/data/fake_image_detector`
11. Go to `Settings` -> `Networking`.
12. Click `Generate Domain`.
13. Open the generated domain when the deploy turns healthy.

### Why the Volume matters

Without a Railway Volume:

- uploaded images
- generated Grad-CAM files
- prediction history

will be runtime-local and may disappear on redeploy or restart.

Railway Volumes provide persistent storage mounted at a directory you choose. Railway documents this in their Volumes guide.

## Deployment Notes

- The trained model file must be present at `model/fake_image_detector.pth`.
- The dashboard reads training metrics from `static/generated/training_metrics.json`.
- For production-grade persistent analytics, replace JSON history with a database.

## Submission Checklist

- App runs locally
- Model file is present
- Training metrics file is present
- GitHub repository is up to date
- README is included
- Demo script is prepared
- Railway config is included

## Important Files

- `app.py`: Flask routes and runtime behavior
- `model/train_model.py`: training pipeline
- `model/predict.py`: inference logic
- `utils/gradcam_visualization.py`: visual explanation generation
- `templates/`: frontend pages
- `static/css/style.css`: styling
- `static/js/app.js`: preview, animations, and charts
