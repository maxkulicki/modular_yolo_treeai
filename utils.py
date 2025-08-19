# utils.py
import torch
import numpy as np
import cv2
from torchvision.ops import roi_align

def load_yolo_annotations(label_path, image_height, image_width):
    """
    Loads YOLO format annotations and converts them to pixel coordinates (x1, y1, x2, y2).
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert from normalized [0, 1] to pixel values
            x_center_px = x_center * image_width
            y_center_px = y_center * image_height
            width_px = width * image_width
            height_px = height * image_height
            
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            x2 = x_center_px + width_px / 2
            y2 = y_center_px + height_px / 2
            
            annotations.append((class_id, [x1, y1, x2, y2]))
    return annotations

def extract_features_for_boxes(feature_maps, boxes, output_size=(7, 7)):
    """
    Extracts fixed-size feature vectors for a list of bounding boxes using RoIAlign.

    Args:
        feature_maps (list[torch.Tensor]): List of feature maps from the Neck.
                                            Typically ordered from largest to smallest.
        boxes (torch.Tensor): A tensor of bounding boxes, shape [N, 4] in (x1, y1, x2, y2) format.
        output_size (tuple): The spatial size of the output feature map for each RoI.

    Returns:
        torch.Tensor: A tensor of feature vectors, shape [N, C, output_size[0], output_size[1]].
    """
    # For simplicity, we use the highest-resolution feature map from the neck output.
    # A more advanced implementation could choose the map based on box size.
    target_feature_map = feature_maps[0] # The P3 feature map (stride 8)

    # RoIAlign expects boxes as a list of tensors for each image in the batch.
    # Since we process one image at a time, we wrap it in a list.
    box_list = [boxes.to(target_feature_map.device)]
    
    # Calculate the spatial scale factor between the image and the feature map
    img_size = (640, 640) # Assuming a default YOLO input size for scale calculation
    fm_size = target_feature_map.shape[-2:]
    spatial_scale = fm_size[0] / img_size[0]

    # Perform RoIAlign
    aligned_features = roi_align(
        target_feature_map,
        box_list,
        output_size=output_size,
        spatial_scale=spatial_scale,
        aligned=True
    )
    
    # Flatten the spatial dimensions to get one vector per box
    # Shape: [num_boxes, channels * output_size[0] * output_size[1]]
    return torch.flatten(aligned_features, start_dim=1)