import os
import json
from pathlib import Path
import kagglehub

# Download latest version
path = kagglehub.dataset_download("usmanafzaal/strawberry-disease-detection-dataset")

print("Path to dataset files:", path)
# Paths
input_dir = path#Path(r"C:\dataset\train")
output_dir = input_dir / 'labels'  # YOLO labels will be saved in the 'labels' directory

# Create the 'labels' directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Define your class mapping (manually or by extracting from dataset)
class_mapping = {
    "Angular Leafspot": 0,
    "Anthracnose Fruit Rot": 1,
    "Blossom Blight": 2,
    "Gray Mold": 3,
    "Leaf Spot": 4,
    "Powdery Mildew Fruit": 5,
    "Powdery Mildew Leaf": 6,


}

# Function to normalize coordinates
def normalize(value, max_value):
    return value / max_value

# Iterate over all JSON files
for json_file in input_dir.glob("*.json"):
    with open(json_file, "r") as file:
        data = json.load(file)

    # Get image dimensions
    img_width = data["imageWidth"]
    img_height = data["imageHeight"]

    # YOLO label content
    yolo_labels = []

    for shape in data["shapes"]:
        class_name = shape["label"]  # Change this to map class names to class IDs
        points = shape["points"]

        # Normalize polygon points
        normalized_points = [
            (normalize(x, img_width), normalize(y, img_height)) for x, y in points
        ]
        flattened_points = [f"{x:.6f} {y:.6f}" for x, y in normalized_points]

        # Compute bounding box from the polygon
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min = max(min(x_coords), 0)
        y_min = max(min(y_coords), 0)
        x_max = min(max(x_coords), img_width)
        y_max = min(max(y_coords), img_height)

        # Convert to YOLO format
        x_center = normalize((x_min + x_max) / 2, img_width)
        y_center = normalize((y_min + y_max) / 2, img_height)
        width = normalize(x_max - x_min, img_width)
        height = normalize(y_max - y_min, img_height)

        # Add label in YOLOv11-Seg format
        # Map the class name to a class ID
        if class_name in class_mapping:
            class_id = class_mapping[class_name]
        else:
            raise ValueError(f"Unknown class '{class_name}' found in {json_file.name}. Please update the class_mapping.")
        yolo_labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " +
            " ".join(flattened_points)
        )

    # Write YOLO label to file
    label_file = output_dir / (json_file.stem + ".txt")
    with open(label_file, "w") as file:
        file.write("\n".join(yolo_labels))

print("Conversion complete. YOLOv11-Seg labels saved!")

