import torch
import numpy as np
import streamlit as st
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# Daftar nama kelas untuk deteksi
CLASS_NAMES = {1: "healthy", 2: "anthracnose"}

# Fungsi untuk memuat model Faster R-CNN
@st.cache_resource
def load_model(model_path):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Fungsi untuk melakukan prediksi pada gambar
def predict(model, image):
    image_tensor = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions[0]

# Streamlit UI
st.title("Deteksi Anthracnose pada Tanaman")
uploaded_image = st.file_uploader("Unggah gambar tanaman", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    image_draw = image.copy()  # Pastikan image_draw terdefinisi
    draw = ImageDraw.Draw(image_draw)

    model = load_model("model.pth")
    predictions = predict(model, image)

    # Gambar bounding box pada hasil prediksi
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= 0.8:
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(label.item(), "Unknown")
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), f"{class_name}: {score:.2f}", fill="blue")

    # Tampilkan hasil deteksi
    st.image(image_draw, caption="Hasil Deteksi", use_column_width=True)
