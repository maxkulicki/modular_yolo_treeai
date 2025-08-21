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

## Project Files

-   `main.py`: Your primary script for **training** and **evaluating** the modular classifier head.
-   `modular_model.py`: Defines the core `ModularYoloClassifier` class that combines the YOLO backbone and the ML head.
-   `infer_large_image.py`: A standalone script to run sliding-window inference on large images using your **modular** model.
-   `infer_yolo_end_to_end.py`: A standalone script to run sliding-window inference using a **standard end-to-end** YOLO model (for baseline comparison).
-   `evaluation.py`: Contains the logic for calculating end-to-end mAP metrics.
-   `feature_extractor.py`: Manages loading the YOLO model and extracting its intermediate features.

## Usage & Workflow

### 1. Setup

First, install the required packages and prepare your data in standard YOLO format (`train/` and `val/` directories with `images/` and `labels/` subfolders).

```bash
pip install ultralytics torch torchvision scikit-learn numpy opencv-python xgboost joblib tqdm
```

### 2. Training & Evaluation (`main.py`)

Use `main.py` to train a new classifier head and evaluate its performance. Configure the switches at the top of the file to control the process.

**To Train a New Classifier:**
1.  Open `main.py`.
2.  Set `CLASSIFIER_TYPE` to `'RandomForest'`, `'XGBoost'`, or `'SVM'`.
3.  Set `RUN_TRAINING = True`.
4.  Run the script: `python main.py`
5.  This will create a saved model file (e.g., `randomforest_classifier.joblib`).

**To Evaluate a Trained Classifier:**
1.  Open `main.py`.
2.  Make sure `CLASSIFIER_TYPE` matches the model you want to evaluate.
3.  Set `RUN_TRAINING = False` and `RUN_EVALUATION = True`.
4.  Run the script: `python main.py`
5.  This produces a detailed report in `evaluation_results.txt` with both classifier accuracy and full-pipeline mAP scores.
