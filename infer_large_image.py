import os
import cv2
import numpy as np
import torch
import argparse
from collections import Counter
from torchvision.ops import nms
from tqdm import tqdm

# Ensure the modular_model script is in the same directory or in the python path
from modular_model import ModularYoloClassifier

def infer_on_large_image(model, img_path, class_names, output_path, tile_size, overlap, conf_threshold, nms_iou_threshold):
    """
    Runs inference on a single large image by tiling it, performing predictions on each tile,
    and merging the results using Non-Max Suppression (NMS).

    Args:
        model (ModularYoloClassifier): The trained modular model instance.
        img_path (str): Path to the large input image.
        class_names (list): A list of strings for the names of the classes.
        output_path (str): Path to save the final visualized image.
        tile_size (int): The size of the square tiles to process.
        overlap (int): The pixel overlap between adjacent tiles.
        conf_threshold (float): Confidence threshold for initial YOLO detections.
        nms_iou_threshold (float): IoU threshold for Non-Max Suppression.
    """
    print("="*50)
    print(f"Starting inference on large image: {os.path.basename(img_path)}")
    
    large_image = cv2.imread(img_path)
    if large_image is None:
        print(f"Error: Could not read image at {img_path}")
        return

    img_h, img_w, _ = large_image.shape
    step = tile_size - overlap
    
    all_detections = []
    
    # Generate a grid of tile coordinates
    tile_coords = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            tile_coords.append((x_start, y_start, x_end, y_end))

    print(f"Generated {len(tile_coords)} tiles of size {tile_size}x{tile_size} for processing...")
    
    # Process each tile
    temp_tile_path = "temp_tile_for_inference.jpg"
    for x1, y1, x2, y2 in tqdm(tile_coords, desc="Processing Tiles"):
        tile = large_image[y1:y2, x1:x2]
        
        # The existing predict function expects a file path, so we save a temporary file
        cv2.imwrite(temp_tile_path, tile)
        
        tile_predictions = model.predict(temp_tile_path, conf_threshold=conf_threshold)
        
        # Adjust box coordinates to the original image space
        for pred in tile_predictions:
            box_local = pred['box']
            box_global = [box_local[0] + x1, box_local[1] + y1, box_local[2] + x1, box_local[3] + y1]
            all_detections.append({'box': box_global, 'species_id': pred['species_id'], 'confidence': pred['confidence']})

    if os.path.exists(temp_tile_path):
        os.remove(temp_tile_path)
    print("✅ Tile processing complete.")
    print(f"   - Found {len(all_detections)} total detections before merging.")

    # Merge overlapping results using Non-Max Suppression (NMS) on a per-class basis
    final_detections = []
    unique_class_ids = set(d['species_id'] for d in all_detections)
    
    for class_id in unique_class_ids:
        class_detections = [d for d in all_detections if d['species_id'] == class_id]
        if not class_detections: continue
        
        boxes_tensor = torch.tensor([d['box'] for d in class_detections], dtype=torch.float32)
        scores_tensor = torch.tensor([d['confidence'] for d in class_detections], dtype=torch.float32)
        
        keep_indices = nms(boxes_tensor, scores_tensor, nms_iou_threshold)
        
        for idx in keep_indices:
            final_detections.append(class_detections[idx])

    print(f"✅ Merging complete. Kept {len(final_detections)} final detections after NMS.")
    
    # Generate Visualization
    print("Generating visualization...")
    num_classes = len(class_names)
    colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in range(num_classes)]
    vis_image = large_image.copy()
    
    for detection in final_detections:
        box, class_id, confidence = detection['box'], int(detection['species_id']), detection['confidence']
        x1, y1, x2, y2 = map(int, box)
        
        class_name = class_names[class_id] if class_id < num_classes else f"Class {class_id}"
        color = colors[class_id] if class_id < num_classes else (255, 255, 255)
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}: {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(vis_image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.imwrite(output_path, vis_image)
    print(f"✅ Visualization saved to {output_path}")

    # Generate Summary Report
    print("\n" + "-"*30)
    print("      Detection Summary Report")
    print("-"*30)
    class_counts = Counter(d['species_id'] for d in final_detections)
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"   - {class_name:<20}: {count} trees")
    print(f"\n   {'TOTAL':<20}: {len(final_detections)} trees")
    print("-"*30)
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run sliding window inference on a large image.")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the large input image.")
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help="Path to the YOLO .pt model file.")
    parser.add_argument('--classifier-type', type=str, required=True, choices=['RandomForest', 'XGBoost', 'SVM'], help="Type of the classifier head used.")
    parser.add_argument('--classifier-model', type=str, required=True, help="Path to the saved .joblib classifier model.")
    parser.add_argument('--class-names', type=str, required=True, help="Comma-separated list of class names, in order of class ID (e.g., 'Pine,Oak,Maple').")
    parser.add_argument('--output-path', type=str, default='output_large_visualization.jpg', help="Path to save the final visualized image.")
    parser.add_argument('--tile-size', type=int, default=640, help="The size of the tiles for inference.")
    parser.add_argument('--overlap', type=int, default=100, help="Pixel overlap between adjacent tiles.")
    parser.add_argument('--conf-threshold', type=float, default=0.25, help="Confidence threshold for YOLO object detection.")
    parser.add_argument('--nms-iou-threshold', type=float, default=0.45, help="IoU threshold for Non-Max Suppression.")
    
    args = parser.parse_args()

    # Convert comma-separated class names to a list
    class_names_list = [name.strip() for name in args.class_names.split(',')]
    
    # Initialize the modular model
    print("Initializing model...")
    model = ModularYoloClassifier(
        yolo_model_path=args.yolo_model,
        classifier_type=args.classifier_type
    )

    # Load the pre-trained classifier head
    try:
        model.load_classifier(args.classifier_model)
    except FileNotFoundError as e:
        print(e)
        exit()

    # Run the main inference function
    infer_on_large_image(
        model=model,
        img_path=args.image_path,
        class_names=class_names_list,
        output_path=args.output_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf_threshold,
        nms_iou_threshold=args.nms_iou_threshold
    )