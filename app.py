import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import HybridPneumoniaCNN, load_trained_model, gradcam_for_images



st.set_page_config(
    page_title="Pneumonia Classifier",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
# Pneumonia Detection — Educational Demo

This tool uses a deep learning model to classify chest X-rays as **NORMAL** or **PNEUMONIA**, and provides interactive Grad-CAM visualizations for model interpretability.

**Important:**  
This application is strictly for **educational purposes**.  
It must **not** be used for clinical diagnosis or medical decision-making.
""")


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
WEIGHTS_PATH = "best_hybrid_pneumonia_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping for available Grad-CAM layers
LAYER_MAP = {
    "Stem Layer": "stem",
    "Residual Block 1": "layer1",
    "Residual Block 2": "layer2",
    "Residual Block 3 (Deepest)": "layer3"
}


@st.cache_resource
def load_model():
    model = HybridPneumoniaCNN(num_classes=2).to(DEVICE)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return img, tensor



uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    
    img, img_tensor = preprocess_image(uploaded_file)
    img_tensor = img_tensor.to(DEVICE)

    col_img, col_pred = st.columns([1, 1])

    with col_img:
        st.subheader("Uploaded Image")
        st.image(img, use_column_width=True)

    
    with col_pred:
        st.subheader("Model Prediction")

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()

        pred_class = CLASS_NAMES[np.argmax(probs)]
        prob_normal = probs[0]
        prob_pneumonia = probs[1]

        st.markdown(f"### Predicted Class: **{pred_class}**")

        # Confidence Indicators
        st.write("### Confidence:")
        st.write(f"NORMAL: {100 * prob_normal:.4f}")
        st.progress(float(prob_normal))

        st.write(f"PNEUMONIA: {100 * prob_pneumonia:.4f}")
        st.progress(float(prob_pneumonia))

    
    st.subheader("Grad-CAM Visualization")

    selected_layer_name = st.selectbox(
        "Choose a layer for interpretability:",
        list(LAYER_MAP.keys())
    )
    selected_layer = LAYER_MAP[selected_layer_name]

    
    if selected_layer == "stem":
        model.gradcam_target_layer = model.stem[0]
    elif selected_layer == "layer1":
        model.gradcam_target_layer = model.layer1[-1].bn2
    elif selected_layer == "layer2":
        model.gradcam_target_layer = model.layer2[-1].bn2
    elif selected_layer == "layer3":
        model.gradcam_target_layer = model.layer3[-1].bn2

    img_np, cam, pred = gradcam_for_images(model, img_tensor, DEVICE, CLASS_NAMES)

    
    col_a, col_b, col_c = st.columns(3)

    # Original
    with col_a:
        st.markdown("### Original Image")
        st.image(img, use_column_width=True)

    # Heatmap Only
    with col_b:
        st.markdown("### Heatmap")
        fig_hm, ax_hm = plt.subplots(figsize=(4,4))
        ax_hm.imshow(cam, cmap="jet")
        ax_hm.axis("off")
        st.pyplot(fig_hm)

    # Overlay
    with col_c:
        st.markdown("### Overlay")
        fig_ov, ax_ov = plt.subplots(figsize=(4,4))
        ax_ov.imshow(img_np)
        ax_ov.imshow(cam, cmap="jet", alpha=0.4)
        ax_ov.axis("off")
        st.pyplot(fig_ov)

    
    st.markdown("""
    <div style='background-color:#263238; padding:14px; border-radius:8px; border:1px solid #90c2e7;'>
    <b>Reminder:</b> Grad-CAM heatmaps highlight areas of model attention, not areas of disease.  
    This tool is intended for educational insight only and must not be used for medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Upload a chest X-ray image to begin.", icon="ℹ️")
