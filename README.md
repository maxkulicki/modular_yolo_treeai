# Modular YOLO: A Framework for Flexible Tree Species Classification

This repository provides a framework for building a flexible tree species classification system that decouples tree detection from species identification.

## Motivation

Standard end-to-end detection models are monolithic. Adding new species or deploying in a new region requires retraining the entire, complex deep learning model. This project demonstrates a more modular and efficient approach.

The core idea is to use a powerful YOLO model as a universal **feature extractor**, and then train a separate, lightweight machine learning model (like RandomForest or XGBoost) as a **swappable classification head**.

This design allows you to:
- **Adapt to new regions quickly:** Train a new species classifier for a specific location in minutes using a small local dataset.
- **Maintain a stable core model:** The powerful YOLO feature extractor is trained once and rarely needs to be touched.
- **Experiment rapidly:** Easily swap between different classifiers (RandomForest, SVM, XGBoost) to find the best head for your specific task.

## How It Works

1.  **Feature Extraction:** A pre-trained YOLO model (e.g., YOLOv8, YOLOv11) processes an image and generates rich, multi-scale feature maps from its intermediate layers (the "neck").
2.  **Training:** For each ground-truth bounding box, we extract a corresponding feature vector from the YOLO feature maps using `RoIAlign`. These vectors are used to train a simple, fast classifier (like RandomForest), which is then saved to disk.
3.  **Inference:** On a new image, YOLO proposes bounding boxes for potential trees. For each box, we extract its feature vector and use our trained lightweight classifier to predict the final species.

## How to Use

### 1. Setup

Install the required packages:
```bash
pip install ultralytics torch scikit-learn numpy opencv-python xgboost joblib tqdm