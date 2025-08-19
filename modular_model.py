import os
import torch
import cv2
import numpy as np
import joblib
from tqdm import tqdm

# --- NEW: Import all the classifiers we want to support ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_extractor import YoloFeatureExtractor
from utils import load_yolo_annotations, extract_features_for_boxes
from ultralytics import YOLO


class ModularYoloClassifier:
    def __init__(self, yolo_model_path='yolov8n.pt', classifier_type='RandomForest'): # <-- MODIFIED
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.feature_extractor = YoloFeatureExtractor(yolo_model_path).to(self.device)
        self.detector = YOLO(yolo_model_path)
        
        # --- NEW: Model factory based on classifier_type ---
        print(f"Initializing classifier of type: {classifier_type}")
        if classifier_type == 'RandomForest':
            self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        
        elif classifier_type == 'XGBoost':
            # XGBoost can use the GPU if available
            self.classifier = XGBClassifier(
                n_estimators=100, 
                eval_metric='mlogloss', 
                tree_method='hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                random_state=42
            )
            
        elif classifier_type == 'SVM':
            # probability=True is required for .predict_proba(), but makes it slower to train.
            self.classifier = SVC(probability=True, random_state=42)
            
        else:
            raise ValueError(f"Unsupported classifier type: '{classifier_type}'. "
                             "Supported types are 'RandomForest', 'XGBoost', 'SVM'.")
        
        print(f"✅ Modular YOLO Classifier with {classifier_type} head initialized.")

    def _prepare_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        return img, img_tensor.unsqueeze(0).to(self.device)

    def _extract_features_from_dataset(self, image_dir, label_dir):
        """Helper function to extract all features and labels from a dataset directory."""
        all_features = []
        all_labels = []
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Extracting features from {os.path.basename(image_dir)}"):
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

            if not os.path.exists(label_path):
                continue

            original_img, img_tensor = self._prepare_image(img_path)
            h, w, _ = original_img.shape

            feature_maps = self.feature_extractor(img_tensor)
            annotations = load_yolo_annotations(label_path, h, w)
            if not annotations:
                continue

            gt_labels = [ann[0] for ann in annotations]
            gt_boxes = torch.tensor([ann[1] for ann in annotations], dtype=torch.float32)

            box_features = extract_features_for_boxes(feature_maps, gt_boxes)
            
            all_features.append(box_features.cpu().numpy())
            all_labels.extend(gt_labels)
        
        if not all_features:
            return None, None
            
        return np.concatenate(all_features, axis=0), np.array(all_labels)

    def train(self, image_dir, label_dir, save_path='rf_classifier.joblib'):
        """Trains the classifier and saves it to a file."""
        X_train, y_train = self._extract_features_from_dataset(image_dir, label_dir)

        if X_train is None:
            print("❌ No features were extracted from the training set. Aborting training.")
            return

        print(f"\n--- Training Classifier ---")
        print(f"   - Training on {X_train.shape[0]} samples.")
        print(f"   - Feature vector dimension: {X_train.shape[1]}")
        
        self.classifier.fit(X_train, y_train)
        print("✅ Classifier training complete.")

        # Save the trained classifier to disk
        joblib.dump(self.classifier, save_path)
        print(f"✅ Classifier saved to {save_path}")

    def evaluate(self, image_dir, label_dir):
            """
            Evaluates the classifier on GT boxes and returns the results as a formatted string.
            """
            print(f"\n--- Evaluating Classifier on {os.path.basename(os.path.dirname(image_dir))} set ---")
            X_eval, y_true = self._extract_features_from_dataset(image_dir, label_dir)

            if X_eval is None:
                msg = "❌ No features were extracted from the evaluation set. Cannot evaluate."
                print(msg)
                return msg
                
            y_pred = self.classifier.predict(X_eval)
            
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            # --- MODIFICATION: Build a results string to return ---
            results_str = f"Overall Accuracy: {accuracy:.4f}\n\n"
            results_str += "Classification Report:\n"
            results_str += report + "\n\n"
            results_str += "Confusion Matrix:\n"
            results_str += str(cm) + "\n"
            
            # Still print to console for immediate feedback
            print(results_str)
            
            return results_str

    def load_classifier(self, path='rf_classifier.joblib'):
        """Loads a pre-trained classifier from a file."""
        print(f"\n--- Loading Classifier from {path} ---")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier file not found at {path}. Please train the model first.")
        self.classifier = joblib.load(path)
        print("✅ Classifier loaded successfully.")

    def predict(self, img_path, conf_threshold=0.4):
        """Performs inference on a new image using the trained/loaded classifier."""
        # This function remains largely the same
        print(f"\n--- Running Inference on {os.path.basename(img_path)} ---")
        original_img, img_tensor = self._prepare_image(img_path)
        
        # self.detector.predictor.args.verbose = False
        # results = self.detector(original_img) 
        results = self.detector(original_img, conf=conf_threshold, verbose=False)
        detected_boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        high_conf_boxes = detected_boxes#[confidences > conf_threshold]

        if high_conf_boxes.shape[0] == 0:
            print("   - No objects detected with confidence >", conf_threshold)
            return []

        print(f"   - Detected {high_conf_boxes.shape[0]} potential objects.")
        
        feature_maps = self.feature_extractor(img_tensor)
        box_features = extract_features_for_boxes(feature_maps, high_conf_boxes).cpu().numpy()
        
        predicted_labels = self.classifier.predict(box_features)
        predicted_probs = self.classifier.predict_proba(box_features)

        print("✅ Inference complete.")
        
        final_predictions = []
        for i, box in enumerate(high_conf_boxes):
            final_predictions.append({
                'box': box.cpu().numpy().tolist(),
                'species_id': predicted_labels[i],
                'confidence': np.max(predicted_probs[i])
            })
        return final_predictions
    
    def generate_predictions_for_map(self, image_dir, output_dir, conf_threshold=0.25):
        """
        Runs the full pipeline on a directory of images and saves predictions
        in YOLO .txt format for mAP calculation.
        """
        print(f"\n--- Generating predictions for mAP evaluation ---")
        os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

        for img_file in tqdm(image_files, desc="Generating predictions"):
            img_path = os.path.join(image_dir, img_file)
            
            # Use the existing predict method to get results for one image
            predictions = self.predict(img_path, conf_threshold=conf_threshold)

            # Convert predictions to YOLO format string
            output_str = ""
            img_h, img_w, _ = cv2.imread(img_path).shape
            for pred in predictions:
                # Convert (x1, y1, x2, y2) back to (x_c, y_c, w, h) normalized
                box = pred['box']
                x1, y1, x2, y2 = box
                
                box_w = x2 - x1
                box_h = y2 - y1
                x_c = x1 + box_w / 2
                y_c = y1 + box_h / 2
                
                # Normalize
                x_c /= img_w
                y_c /= img_h
                box_w /= img_w
                box_h /= img_h
                
                class_id = pred['species_id']
                # Using confidence from the RF classifier, not the detector
                confidence = pred['confidence']
                
                output_str += f"{class_id} {x_c} {y_c} {box_w} {box_h} {confidence}\n"

            # Write to file
            label_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(output_str)

        # Re-enable verbose printing for single-image predictions if needed
        self.detector.predictor.args.verbose = True
        print(f"\n✅ Predictions for {len(image_files)} images saved to {output_dir}")