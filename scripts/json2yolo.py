import os
import cv2
import numpy as np
from pathlib import Path

# Download dataset
#import kagglehub
#path = kagglehub.dataset_download("usmanafzaal/strawberry-disease-detection-dataset")
#print("Path to dataset files:", path)
path= 'c:/dataset'
# Paths
dataset_path = Path(path)
images_path = dataset_path / "train"  # Update with your dataset's image folder
masks_path = dataset_path / "train"    # Update with your dataset's mask folder

output_path = Path("yolo_segmentation")
output_images = output_path / "images"
output_labels = output_path / "labels"

# Create directories
(output_images / "train").mkdir(parents=True, exist_ok=True)
(output_images / "val").mkdir(parents=True, exist_ok=True)
(output_images / "test").mkdir(parents=True, exist_ok=True)
(output_labels / "train").mkdir(parents=True, exist_ok=True)
(output_labels / "val").mkdir(parents=True, exist_ok=True)
(output_labels / "test").mkdir(parents=True, exist_ok=True)

# Convert dataset
for mask_file in masks_path.glob("*.png"):  # Update extension if needed
    image_name = mask_file.stem + ".jpg"    # Update extension if needed
    image_file = images_path / image_name

    # Read mask and image
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_file))
    h, w, _ = image.shape

    # Find contours (for segmentation mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_label = []

    for contour in contours:
        # Bounding box
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        x_center = (x + bbox_w / 2) / w
        y_center = (y + bbox_h / 2) / h
        width = bbox_w / w
        height = bbox_h / h

        # Polygon points (normalized)
        polygon = []
        for point in contour:
            px, py = point[0]
            polygon.append(px / w)
            polygon.append(py / h)
        
        # YOLOv11-Seg label
        label = [0, x_center, y_center, width, height] + polygon
        yolo_label.append(" ".join(map(str, label)))

    # Save label
    label_file = output_labels / "train" / (mask_file.stem + ".txt")  # Adjust for train/val/test
    with open(label_file, "w") as f:
        f.write("\n".join(yolo_label))

    # Copy image to output directory
    output_image_file = output_images / "train" / image_name  # Adjust for train/val/test
    cv2.imwrite(str(output_image_file), image)

print("YOLOv11-Seg dataset preparation complete!")
