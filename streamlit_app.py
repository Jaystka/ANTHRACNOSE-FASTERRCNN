import torch
import streamlit as st
import numpy as np
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image
import io
import base64

CLASS_NAMES = {1: "healthy", 2: "anthracnose"}

# Fungsi untuk memuat model Faster R-CNN
@st.cache_resource
def load_model(model_path):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=3)
    model.load_state_dict(torch.load("fasterrcnn_anthracnose_detector13.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# Fungsi untuk prediksi
def predict(model, image):
    image_tensor = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions[0]

# Streamlit UI
st.title("Deteksi Anthracnose pada Tanaman")
model = load_model("fasterrcnn_anthracnose_detector13.pth")

# HTML untuk akses kamera
st.markdown(
    """
    <div style="text-align: center;">
        <video id="video" width="100%" autoplay></video>
        <button id="capture" onclick="capturePhoto()">Ambil Foto</button>
        <canvas id="canvas" style="display: none;"></canvas>
    </div>

    <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');

    // Akses kamera
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
      })
      .catch((error) => {
        console.error("Error accessing camera:", error);
      });

    function capturePhoto() {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      const dataUrl = canvas.toDataURL('image/png');
      const base64Image = dataUrl.split(',')[1];
      fetch('/upload-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image }),
      }).then(response => response.json())
        .then(data => {
          if (data.prediction) {
            Streamlit.setComponentValue(data.prediction);
          }
        });
    }
    </script>
    """,
    unsafe_allow_html=True,
)

# Fungsi untuk menangani gambar yang diunggah
def process_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    predictions = predict(model, image)
    
    return predictions

# Baca input dari JavaScript
if "image_data" in st.session_state:
    base64_image = st.session_state.image_data
    predictions = process_image(base64_image)
    # Tampilkan hasil deteksi pada gambar yang diambil
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= 0.8:
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(label.item(), "Unknown")
            image_draw = image.copy()
            draw = ImageDraw.Draw(image_draw)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), f"{class_name}: {score:.2f}", fill="blue")

    st.image(image_draw, caption="Hasil Deteksi", use_column_width=True)
