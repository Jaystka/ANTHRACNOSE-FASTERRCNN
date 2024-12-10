import torch
import numpy as np
import streamlit as st
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import time  # Untuk menghitung waktu deteksi
import torchmetrics

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

# Fungsi untuk menghitung mAP
def calculate_mAP(predictions, ground_truths):
    iou_threshold = 0.5  # Threshold IoU untuk mAP
    # Inisialisasi objek untuk menghitung metrik
    metric = torchmetrics.detection.MeanAveragePrecision()
    # Menambahkan prediksi dan ground truth ke metric
    metric.update(predictions, ground_truths)
    # Menghitung mAP
    mAP = metric.compute()
    return mAP

# Streamlit UI
st.title("Deteksi Anthracnose pada Pisang Algoritma FASTER R CNN")
uploaded_image = st.file_uploader("Unggah gambar tanaman", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)

    # Pilih font dan ukuran dasar
    try:
        font = ImageFont.truetype("arial.ttf", size=40)  # Ukuran dasar untuk font
    except IOError:
        font = ImageFont.load_default()  # Gunakan font default jika font tidak ditemukan

    # Muat model dan hitung waktu deteksi
    model = load_model("fasterrcnn_anthracnose_detector33.pth")
    start_time = time.time()  # Mulai penghitungan waktu
    predictions = predict(model, image)
    end_time = time.time()  # Selesai penghitungan waktu

    detection_time = end_time - start_time  # Waktu deteksi

    # Gambar bounding box pada hasil prediksi
    ground_truths = []  # Ganti dengan ground truth jika ada
    predicted_boxes = []
    predicted_labels = []
    predicted_scores = []
    
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= 0.3:
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(label.item(), "Unknown")
            
            # Tentukan ukuran font berdasarkan lebar bounding box
            box_width = x2 - x1
            box_height = y2 - y1
            font_size = 40  # Menyesuaikan ukuran font dengan lebar box
            
            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except IOError:
                font = ImageFont.load_default()  # Gunakan font default jika font tidak ditemukan

            # Gambar kotak putih di bawah label
            text_bbox = draw.textbbox((x1, y1), f"{class_name}: {score:.2f}", font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            margin = 5  # Jarak margin antara teks dan kotak
            draw.rectangle([x1, y1 - text_height - margin, x1 + text_width + margin, y1], fill="red")  # Kotak putih
            
            # Gambar teks dengan ukuran font baru
            draw.text((x1 + margin, y1 - text_height - margin), f"{class_name}: {score:.2f}", fill="white", font=font)

            # Gambar bounding box (kotak merah)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

            # Menyimpan hasil prediksi untuk perhitungan mAP
            predicted_boxes.append([x1, y1, x2, y2])
            predicted_labels.append(label.item())
            predicted_scores.append(score.item())
            # Menambahkan ground truth jika ada
            ground_truths.append([x1, y1, x2, y2])  # Misalnya ground truth ada di sini
    
    # Menghitung mAP
    if len(predicted_boxes) > 0:
        # Struktur format deteksi dan ground truth untuk mAP
        predictions_dict = [{
            'boxes': torch.tensor(predicted_boxes),
            'labels': torch.tensor(predicted_labels),
            'scores': torch.tensor(predicted_scores)
        }]
        
        ground_truths_dict = [{
            'boxes': torch.tensor(ground_truths),
            'labels': torch.tensor(predicted_labels)  # Gantilah dengan label ground truth
        }]
        
        mAP = calculate_mAP(predictions_dict, ground_truths_dict)
        st.write(f"**mAP@0.5:** {mAP['map']:.2f}")
    
    # Tampilkan hasil deteksi
    st.image(image_draw, caption="Hasil Deteksi", use_container_width=True)

    # Tampilkan waktu deteksi
    st.write(f"**Waktu Deteksi:** {detection_time:.2f} detik")
