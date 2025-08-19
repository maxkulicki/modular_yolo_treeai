
# main.py
import os
from modular_model import ModularYoloClassifier
import cv2
from evaluation import calculate_map_metrics # <-- NEW IMPORT
from datetime import datetime

# --- Configuration ---

CLASSIFIER_TYPE = 'RandomForest'  # Options: 'RandomForest', 'XGBoost', 'SVM'

# Set these flags to control the workflow
RUN_TRAINING = True
RUN_EVALUATION = True  # Master switch for all evaluations
RUN_SINGLE_IMAGE_INFERENCE = True

# --- Paths ---
TRAIN_DIR = '/home/makskulicki/treeAI/data/12_RGB_FullyLabeled_640_NewSplit/train'
VAL_DIR = '/home/makskulicki/treeAI/data/12_RGB_FullyLabeled_640_NewSplit/val'
TEST_IMAGE_PATH = '/home/makskulicki/treeAI/data/12_RGB_FullyLabeled_640_NewSplit/test/images/000000000094.png' # Make sure this path is correct
YOLO_MODEL_PATH = '/home/makskulicki/treeAI/treeAI/y11l_dataset12_conifer_broadleaf2/weights/best.pt' # Use a pretrained model or your custom trained one
MAP_PRED_DIR = '/home/makskulicki/treeAI/yolo_modular_classifier/predictions/12_val_species/pred_labels' # Directory to save predictions for mAP
EVALUATION_OUTPUT_PATH = f'/home/makskulicki/treeAI/yolo_modular_classifier/predictions/{CLASSIFIER_TYPE.lower()}_evaluation_results.txt'

CLASSIFIER_SAVE_PATH = f"/home/makskulicki/treeAI/yolo_modular_classifier/trained_classifiers/species/{CLASSIFIER_TYPE.lower()}_classifier.joblib"

# --- Main Execution ---
if __name__ == '__main__':
    train_image_dir, train_label_dir = os.path.join(TRAIN_DIR, 'images'), os.path.join(TRAIN_DIR, 'labels')
    val_image_dir, val_label_dir = os.path.join(VAL_DIR, 'images'), os.path.join(VAL_DIR, 'labels')
    
    # --- MODIFIED: Pass the classifier type during initialization ---
    model = ModularYoloClassifier(
        yolo_model_path=YOLO_MODEL_PATH,
        classifier_type=CLASSIFIER_TYPE
    )
    
    # --- Training or Loading ---
    if RUN_TRAINING:
        model.train(train_image_dir, train_label_dir, CLASSIFIER_SAVE_PATH)
    else:
        try:
            model.load_classifier(CLASSIFIER_SAVE_PATH)
        except FileNotFoundError as e:
            print(f"Error: Could not find saved model at '{CLASSIFIER_SAVE_PATH}'.")
            print("Please set RUN_TRAINING to True to train a new model first.")
            exit()

    # --- Combined Evaluation Block (no changes needed here) ---
    if RUN_EVALUATION and os.path.exists(val_image_dir):
        # ... (This entire block remains the same as before) ...
        # It will automatically use the selected classifier.
        all_results_content = ""
        print("\n" + "="*50); print("Running CLASSIFIER evaluation (on ground-truth boxes)...")
        classifier_results = model.evaluate(val_image_dir, val_label_dir)
        all_results_content += "--- CLASSIFIER PERFORMANCE (on Ground-Truth Boxes) ---\n"
        all_results_content += classifier_results + "\n\n"
        print("="*50)

        print("\n" + "="*50); print("Running full PIPELINE mAP evaluation...")
        model.generate_predictions_for_map(val_image_dir, MAP_PRED_DIR)
        map_results = calculate_map_metrics(gt_dir=val_label_dir, pred_dir=MAP_PRED_DIR)
        all_results_content += "--- END-TO-END PIPELINE PERFORMANCE (mAP) ---\n"
        if map_results:
            all_results_content += f"mAP @ .50          : {map_results['mAP@.5']:.4f}\n"
            all_results_content += f"mAP @ .50:.95 (COCO): {map_results['mAP@.5:.95']:.4f}\n"
        else:
            all_results_content += "mAP calculation failed. Check logs.\n"
        print("="*50)

        with open(EVALUATION_OUTPUT_PATH, 'w') as f:
            f.write(f"Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"YOLO Model: {YOLO_MODEL_PATH}\n")
            f.write(f"Classifier Type: {CLASSIFIER_TYPE}\n") # Added for clarity
            f.write(f"Classifier File: {CLASSIFIER_SAVE_PATH}\n") # Added for clarity
            f.write("="*50 + "\n\n")
            f.write(all_results_content)
        
        print(f"\n✅ All evaluation results have been saved to: {EVALUATION_OUTPUT_PATH}")

    # --- Single Image Inference ---
    if RUN_SINGLE_IMAGE_INFERENCE and os.path.exists(TEST_IMAGE_PATH):
        # ... (This block remains unchanged) ...
        print("\n" + "="*50)
        print("Running single image inference...")
        predictions = model.predict(img_path=TEST_IMAGE_PATH)
        print("\n--- Predictions ---")
        if predictions:
            test_img = cv2.imread(TEST_IMAGE_PATH)
            for pred in predictions:
                box = pred['box']; label = f"Species: {pred['species_id']} ({pred['confidence']:.2f})"
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(test_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            output_path = 'output_predictions.jpg'
            cv2.imwrite(output_path, test_img)
            print(f"\n✅ Saved visualization to {output_path}")
        print("="*50)