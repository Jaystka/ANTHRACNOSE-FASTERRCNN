import torch
import numpy as np
import streamlit as st
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import time  # Untuk menghitung waktu deteksi

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
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)

    # Pilih font dan ukuran
    try:
        font = ImageFont.truetype("arial.ttf", size=40)  # Anda bisa mengganti ukuran sesuai kebutuhan
    except IOError:
        font = ImageFont.load_default()  # Gunakan font default jika font tidak ditemukan

    # Muat model dan hitung waktu deteksi
    model = load_model("fasterrcnn_anthracnose_detector33.pth")
    start_time = time.time()  # Mulai penghitungan waktu
    predictions = predict(model, image)
    end_time = time.time()  # Selesai penghitungan waktu

    detection_time = end_time - start_time  # Waktu deteksi

    # Gambar bounding box pada hasil prediksi
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= 0.8:
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(label.item(), "Unknown")
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            # Gambar teks dengan ukuran font besar
            draw.text((x1, y1 - 20), f"{class_name}: {score:.2f}", fill="white", font=font)

    # Tampilkan hasil deteksi
    st.image(image_draw, caption="Hasil Deteksi", use_container_width=True)

    # Tampilkan waktu deteksi
    st.write(f"**Waktu Deteksi:** {detection_time:.2f} detik")
