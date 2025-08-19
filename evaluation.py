# evaluation.py
import os
import numpy as np
from collections import defaultdict

# --- Helper functions (from your script) ---

def yolo_to_bbox_corners(x_c, y_c, w, h):
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
    inter_w, inter_h = max(0.0, inter_x_max - inter_x_min), max(0.0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h
    if intersection <= 0.0: return 0.0
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    return intersection / (union + 1e-16)

def calculate_average_precision(recalls, precisions):
    ap, recalls, precisions = 0.0, np.asarray(recalls), np.asarray(precisions)
    for t in np.linspace(0.0, 1.0, 11):
        p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0.0
        ap += p
    return ap / 11.0

# --- Data Loaders (from your script) ---

def _iter_txt_files(folder, ext=".txt"):
    if not os.path.isdir(folder): raise FileNotFoundError(f"Folder not found: {folder}")
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(ext): yield os.path.join(folder, name)

def load_ground_truth_from_folder(folder):
    gts = []
    for path in _iter_txt_files(folder):
        image_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                    gts.append({"image_id": image_id, "class_id": cls, "box": yolo_to_bbox_corners(x, y, w, h)})
    return gts

def load_predictions_from_folder(folder):
    preds = []
    for path in _iter_txt_files(folder):
        image_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls, x, y, w, h, conf = int(parts[0]), *map(float, parts[1:6])
                    preds.append({"image_id": image_id, "class_id": cls, "box": yolo_to_bbox_corners(x, y, w, h), "confidence": conf})
    return preds

# --- Main mAP Computation Logic (refactored into a function) ---

def compute_map(all_ground_truths, all_predictions, iou_thresholds, class_ids):
    if not all_ground_truths: return 0.0
    gt_by_img_cls = defaultdict(list)
    for gt in all_ground_truths:
        gt_by_img_cls[(gt['image_id'], gt['class_id'])].append({'box': gt['box'], 'detected': False})
    
    all_predictions_sorted = sorted(all_predictions, key=lambda d: d['confidence'], reverse=True)
    
    aps_per_threshold = []
    for thr in iou_thresholds:
        # Reset detection flags for each threshold
        for key in gt_by_img_cls:
            for gt in gt_by_img_cls[key]:
                gt['detected'] = False

        ap_per_class = []
        for cls in class_ids:
            class_preds = [p for p in all_predictions_sorted if p['class_id'] == cls]
            n_gts = sum(len(v) for (img, c), v in gt_by_img_cls.items() if c == cls)
            if n_gts == 0: continue
            
            tp, fp = np.zeros(len(class_preds)), np.zeros(len(class_preds))
            for idx, pred in enumerate(class_preds):
                gt_boxes_for_pred = gt_by_img_cls.get((pred['image_id'], cls), [])
                if not gt_boxes_for_pred:
                    fp[idx] = 1.0
                    continue
                
                ious = [compute_iou(pred['box'], gt['box']) for gt in gt_boxes_for_pred]
                best_gt_idx = np.argmax(ious)
                
                if ious[best_gt_idx] >= thr and not gt_boxes_for_pred[best_gt_idx]['detected']:
                    tp[idx] = 1.0
                    gt_boxes_for_pred[best_gt_idx]['detected'] = True
                else:
                    fp[idx] = 1.0
            
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
            recalls = cum_tp / (n_gts + 1e-16)
            precisions = cum_tp / (cum_tp + cum_fp + 1e-16)
            ap_per_class.append(calculate_average_precision(recalls, precisions))
        
        aps_per_threshold.append(np.mean(ap_per_class) if ap_per_class else 0.0)
    
    return float(np.mean(aps_per_threshold))

# --- The main callable function for our project ---

def calculate_map_metrics(gt_dir, pred_dir):
    """
    Loads GT and prediction files from directories and calculates mAP scores.
    Returns a dictionary of the results.
    """
    print("\n--- Calculating End-to-End Pipeline mAP Metrics ---")
    print("   - Loading ground-truth labels...")
    gts = load_ground_truth_from_folder(gt_dir)
    print(f"   - Loading prediction labels...")
    preds = load_predictions_from_folder(pred_dir)
    
    if not gts or not preds:
        print("   - Ground truth or predictions are empty. Cannot calculate mAP.")
        return None

    class_ids = sorted({gt['class_id'] for gt in gts})
    print(f"   - Found {len(class_ids)} classes in the ground truth.")

    print("   - Computing mAP@0.5...")
    map_50 = compute_map(gts, preds, iou_thresholds=np.array([0.5]), class_ids=class_ids)
    
    print("   - Computing mAP@0.5:0.95...")
    map_50_95 = compute_map(gts, preds, iou_thresholds=np.arange(0.5, 1.0, 0.05), class_ids=class_ids)
    
    # --- MODIFICATION: Return results instead of printing ---
    results = {
        "mAP@.5": map_50,
        "mAP@.5:.95": map_50_95
    }
    
    # Still print a summary to the console for real-time feedback
    print("\n--- mAP Results ---")
    print(f"   mAP@.5         : {results['mAP@.5']:.4f}")
    print(f"   mAP@.50:.95 (COCO): {results['mAP@.5:.95']:.4f}")
    print("-------------------")
    
    return results