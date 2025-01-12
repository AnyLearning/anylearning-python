import yaml
import json
import pickle
import cv2
import numpy as np
from PIL import Image
import torch

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import ColorMode, Visualizer

CONFIG_PATH = "/path/to/config.yml"
DETECTRON2_CONFIG_PATH = "/path/to/detectron2_cfg.pkl"
MODEL_PATH = "/path/to/model.pth"
IMAGE_PATH = "/path/to/image.png"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

with open(DETECTRON2_CONFIG_PATH, "rb") as f:
    detectron2_cfg = pickle.load(f)


class Predictor:
    def __init__(self, cfg, model):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = model
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions
        
# load the model
model = torch.load(MODEL_PATH)
model.eval()        

# create a metadata to map the class id to the class name
class_names = [item["name"] for item in config["data"]["label_set"]]
id_map = {i+1:i for i in range(len(class_names))}
metadata =  Metadata.set(dict(thing_classes=class_names,
                            thing_dataset_id_to_contiguous_id=id_map))

# create a predictor
predictor = Predictor(detectron2_cfg, model)

# read RGB image 
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

outputs = predictor(image_bgr)
instances = outputs["instances"]

v = Visualizer(image_rgb,
                metadata = metadata,
                font_size_scale=2.0,
                instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
)

out_pred = v.draw_instance_predictions(instances.to("cpu"))
visualization_image = out_pred.get_image()

# save the visualization image
cv2.imwrite("visualization.png", visualization_image)

formatted_predictions = dict()
formatted_predictions["boxes"] = instances.pred_boxes.tensor.cpu().numpy().tolist()
formatted_predictions["scores"] = instances.scores.cpu().numpy().tolist()
formatted_predictions["labels"] = [class_names[i] for i in instances.pred_classes.cpu().numpy().tolist()]
formatted_predictions["masks"] = []
# convert masks to polygons
for mask in instances.pred_masks.cpu().numpy().astype('uint8'):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    formatted_predictions["masks"].append([contour.flatten().tolist() for contour in contours])

# save the formatted predictions
with open("formatted_predictions.json", "w") as f:
    json.dump(formatted_predictions, f)