# Demo Script

## 1. Opening

Hello everyone. This project is called **AI Image Authenticity Detector using Deep Learning**.

The goal of the system is to analyze an uploaded image and classify it as either a **Real Image** or an **AI Generated Image**. Along with the prediction, the system also shows the confidence score, an explanation, and a Grad-CAM visualization highlighting the regions that influenced the decision.

## 2. Tech Stack

This project is built with:

- Flask for the web backend
- PyTorch for the deep learning model
- OpenCV for image preprocessing
- Chart.js for confidence visualization
- Grad-CAM for model interpretability
- HTML, CSS, JavaScript, TailwindCSS, and GSAP for the frontend

## 3. Show the Landing Page

On the landing page, we can see a summary of the system, some quick analytics, and navigation to the upload page and dashboard.

Point out:

- modern responsive interface
- total images tested
- real vs fake count
- validation accuracy summary

## 4. Show the Upload Flow

Next, I go to the upload page.

Here the user can:

- select an image file
- preview it before analysis
- run prediction using the trained model

The system only accepts JPG, JPEG, and PNG images.

## 5. Show the Prediction Result

After uploading an image, the model predicts whether it is real or AI generated.

On the result page, I explain:

- the prediction label
- the confidence score
- the probability chart
- the explanation text
- the original image
- the Grad-CAM heatmap
- the overlay comparison

Suggested line:

"The Grad-CAM visualization helps us understand which regions of the image contributed most to the model's decision, making the system more interpretable."

## 6. Show the Dashboard

Now I open the dashboard.

Here we can see:

- total number of images tested
- real vs fake distribution
- training accuracy and loss information
- confusion matrix
- recent prediction history

Suggested line:

"This dashboard combines live usage analytics with model validation metrics, so the project is not just predicting, but also reporting its performance clearly."

## 7. Brief Model Explanation

The model is a CNN trained on real and AI generated images.

During preprocessing, each image is:

- resized to 128 by 128
- normalized
- optionally augmented during training with flips, rotations, and crop-based variation

The trained model is then saved and loaded by the Flask app for inference.

## 8. Closing

In summary, this project demonstrates:

- a complete deep learning workflow
- an end-to-end web application
- interpretable AI output using Grad-CAM
- a clean user interface for non-technical users

Thank you.
