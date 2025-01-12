import yaml
import json

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

CONFIG_PATH = "/path/to/config.yml"
MODEL_PATH = "/path/to/model.pth"
IMAGE_PATH = "/path/to/image.png"


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in range(0, len(hex_color), 2))

def get_transformations(img_size, mean, std):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform

label_set = config["data"]["label_set"]
label_id_to_name = {v["id"]: v["name"] for v in label_set}
label_id_to_color = {v["id"]: hex_to_rgb(v["color"]) for v in label_set}

model = torch.load(MODEL_PATH)
model.eval()
transform = get_transformations(config["data"]["img_size"], 
                                mean=config["data"]["normalize"]["mean"],
                                std=config["data"]["normalize"]["std"])

pil_image = Image.open(IMAGE_PATH).convert('RGB')
inp = transform(pil_image)
inp = inp.unsqueeze(0)
    
with torch.no_grad():
    pred_mask = model(inp)
    pred_mask = torch.argmax(pred_mask, dim=1)[0]

# Visualize
np_image = np.array(pil_image)
pred_mask = pred_mask.cpu().numpy()
pred_mask = pred_mask.astype(np.uint8)
pred_mask = cv2.resize(pred_mask, (np_image.shape[1], np_image.shape[0]), 
                        interpolation=cv2.INTER_NEAREST)

visualization_image = np.zeros_like(np_image)
for label_id, color in label_id_to_color.items():
    visualization_image[pred_mask == label_id] = color

# Blend with original image
alpha = 0.5
visualization_image = cv2.addWeighted(
    np_image, alpha, visualization_image, 1-alpha, 0
)

# save the visualization image
cv2.imwrite("visualization.png", visualization_image)

# formatted predictions, to polygon format
formatted_predictions = []
for label_id, name in label_id_to_name.items():
    mask = (pred_mask == label_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > 5: # filter out small polygons (noise)
            all_polygons.append(contour.flatten().tolist())
    if all_polygons:
        formatted_predictions.append({
            "id": label_id,
            "label": name,
            "color": label_id_to_color[label_id],
            "points": all_polygons,
            "type": "polygon"
        })

# save the formatted predictions
with open("formatted_predictions.json", "w") as f:
    json.dump(formatted_predictions, f)