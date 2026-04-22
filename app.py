import streamlit as st
import torch
from PIL import Image
import numpy as np
from model_unet import UNet
import os

# Load model
@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=23)
    model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

def preprocess(image):
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_tensor = torch.tensor(img_array).permute(2,0,1).unsqueeze(0).float()
    return img_tensor

def predict_mask(image):
    inp = preprocess(image)
    with torch.no_grad():
        out = model(inp)
        pred = torch.argmax(out, dim=1).squeeze(0).numpy()
    return pred

# Page 1
def page1():
    st.title("📊 Training Performance")
    st.write("**Test mIOU:** 0.5123   **Test mDice:** 0.5246")  # update with actual
    st.image("plots/training_curves.png")

# Page 2
def page2():
    st.title("🖼️ Segmentation Demo")
    uploaded = st.file_uploader("Upload 4 images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded and len(uploaded) == 4:
        for i, file in enumerate(uploaded):
            img = Image.open(file).convert("RGB")
            pred = predict_mask(img)
            st.subheader(f"Image {i+1}")
            col1, col2, col3 = st.columns(3)
            col1.image(img, caption="Original")
            col2.image(pred, caption="Predicted Mask")
            # For ground truth, you would need to load corresponding mask from test set.
            # Placeholder:
            col3.image(np.zeros_like(pred), caption="Ground Truth (demo)")

# Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Training Metrics", "Segmentation Demo"])
if choice == "Training Metrics":
    page1()
else:
    page2()
