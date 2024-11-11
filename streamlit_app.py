
import torch
import cv2
import numpy as np
import streamlit as st
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image

# Daftar nama kelas untuk deteksi
CLASS_NAMES = {2: "anthracnose", 1: "healthy"}  # Sesuaikan indeks kelas dengan model Anda

# Fungsi untuk memuat model Faster R-CNN
@st.cache_resource
def load_model(model_path):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Fungsi untuk melakukan prediksi pada frame
def predict(model, frame):
    # Konversi frame ke tensor
    image = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image)
    return predictions[0]

# Fungsi untuk menangkap video secara realtime dari webcam
def run_realtime_detection(model):
    cap = cv2.VideoCapture(0)  # Gunakan 0 untuk webcam pertama
    frame_display = st.empty()  # Tempat kosong untuk memperbarui gambar
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prediksi
        predictions = predict(model, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # Tampilkan bounding box dan label pada frame
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score >= 0.8:  # Threshold
                x1, y1, x2, y2 = map(int, box)
                class_name = CLASS_NAMES.get(label.item(), "Unknown")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Ubah warna frame dari BGR ke RGB dan tampilkan di Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(frame, channels="RGB", use_container_width=True)  # Mengganti gambar sebelumnya

    cap.release()

# Memuat model
model_path = "fasterrcnn_anthracnose_detector13.pth"
model = load_model(model_path)

# Judul aplikasi
st.title("Real-time Detection dengan Faster R-CNN")
st.write("Menggunakan webcam untuk mendeteksi penyakit tanaman secara realtime.")

if st.button("Mulai Deteksi"):
    run_realtime_detection(model)
