import json

# Load the COCO JSON file
with open("/home/edramos/Documents/datasets/Strawberry-Diseases-1/annotations/test.json", "r") as f:
    data = json.load(f)

# Find the category_id for 'Fruits'
fruits_category_id = None
for category in data["categories"]:
    if category["name"] == "Fruits":
        fruits_category_id = category["id"]
        break

# If 'Fruits' category exists, proceed to remove it
if fruits_category_id is not None:
    # Remove annotations linked to 'Fruits'
    data["annotations"] = [ann for ann in data["annotations"] if ann["category_id"] != fruits_category_id]
    
    # Remove 'Fruits' from categories
    data["categories"] = [cat for cat in data["categories"] if cat["id"] != fruits_category_id]

# Save the updated JSON
with open("/home/edramos/Documents/datasets/Strawberry-Diseases-1/annotations/test.json", "w") as f:
    json.dump(data, f, indent=4)

print("Updated JSON saved as 'test.json'")
