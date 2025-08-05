import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Mapping color names to RGB values
COLOR_MAP = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "yellow": [1, 1, 0],
    "purple": [0.5, 0, 0.5],
    "cyan": [0, 1, 1],
    "orange": [1, 0.5, 0],
    "black": [0, 0, 0],
    "white": [1, 1, 1],
    "magenta": [1, 0, 1]  
    
}


def color_name_to_tensor(color_name, H, W):
    if color_name not in COLOR_MAP:
        raise ValueError(f"Unknown color '{color_name}' in COLOR_MAP")
    color = torch.tensor(COLOR_MAP[color_name], dtype=torch.float32).view(3, 1, 1)
    return color.expand(-1, H, W)

class PolygonDataset(Dataset):
    def __init__(self, data_dir):
        self.inputs_dir = os.path.join(data_dir, "inputs")
        self.outputs_dir = os.path.join(data_dir, "outputs")
        json_path = os.path.join(data_dir, "data.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"data.json not found in {data_dir}")
        with open(json_path, 'r') as f:
            self.mapping = json.load(f)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        entry = self.mapping[idx]

        # Ensure required keys are present
        if "input_polygon" not in entry or "colour" not in entry or "output_image" not in entry:
            raise KeyError(f"Missing expected keys in entry: {entry}")

        # Load grayscale polygon input
        input_path = os.path.join(self.inputs_dir, entry["input_polygon"])
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        polygon_image = Image.open(input_path).convert("L")
        polygon = np.array(polygon_image, dtype=np.float32) / 255.0
        polygon_tensor = torch.tensor(polygon).unsqueeze(0)  # Shape: [1, H, W]

        H, W = polygon_tensor.shape[1:]
        color_tensor = color_name_to_tensor(entry["colour"], H, W)  # Shape: [3, H, W]

        # Concatenate grayscale + color condition → [4, H, W]
        input_tensor = torch.cat([polygon_tensor, color_tensor], dim=0)

        # Load expected RGB output
        output_path = os.path.join(self.outputs_dir, entry["output_image"])
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output image not found: {output_path}")
        output_image = Image.open(output_path).convert("RGB")
        output = np.array(output_image, dtype=np.float32) / 255.0
        output_tensor = torch.tensor(output).permute(2, 0, 1)  # Shape: [3, H, W]

        return input_tensor, output_tensor
