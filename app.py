import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import ColorCNN
from data import get_dataloaders

# Hàm tải mô hình
@st.cache_resource
def load_model(model_path, num_classes):
    model = ColorCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Hàm dự đoán
def predict(image, model, transform, classes):
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]

# Thiết lập giao diện
st.markdown(f"<b>Prediction:</b> {predict}", unsafe_allow_html=True)
st.write
