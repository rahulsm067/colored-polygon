import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from unet_model import UNet

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Polygon Colorizer", layout="centered")
st.title("🎨 Polygon Colorizer using UNet")

# --- Color dictionary ---
COLOR_MAP = {
    "red": [255, 0, 0],
    "blue": [0, 0, 255],
    "green": [0, 255, 0],
    "yellow": [255, 255, 0],
    "purple": [128, 0, 128],
    "orange": [255, 165, 0],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
}

# --- Load model ---
@st.cache_resource
def load_model():
    model = UNet(in_channels=4, out_channels=3)
    model.load_state_dict(torch.load("best_unet.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Upload image ---
uploaded_file = st.file_uploader("Upload a polygon image", type=["png", "jpg", "jpeg"])
selected_color = st.selectbox("Select a fill color", list(COLOR_MAP.keys()))

# --- Processing & Prediction ---
if uploaded_file and selected_color:
    input_image = Image.open(uploaded_file).convert("L").resize((128, 128))
    st.image(input_image, caption="Uploaded Polygon", use_container_width=True)

    if st.button("Generate Colored Polygon"):
        # Convert image to float32 tensor
        img_array = np.array(input_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 128]

        # Create color tensor
        color_rgb = torch.tensor(COLOR_MAP[selected_color], dtype=torch.float32).view(3, 1, 1) / 255.0
        color_tensor = color_rgb.expand(3, 128, 128)

        # Combine image and color channel
        input_tensor = torch.cat([img_tensor, color_tensor.unsqueeze(0)], dim=1)  # [1, 4, 128, 128]

        # Model inference
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.clamp(output.squeeze(0), 0, 1).permute(1, 2, 0).numpy()
            output_image = Image.fromarray((output * 255).astype(np.uint8))

        st.image(output_image, caption=" Colored Polygon Output", use_container_width=True)
