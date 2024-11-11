import torch
import numpy as np
import streamlit as st
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image
import base64
import cv2
import io

# Daftar nama kelas untuk deteksi
CLASS_NAMES = {1: "healthy", 2: "anthracnose"}

# Fungsi untuk memuat model Faster R-CNN
@st.cache_resource
def load_model():
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=3)
    model.load_state_dict(torch.load("fasterrcnn_anthracnose_detector13.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# Inisialisasi model
model = load_model()

# Komponen HTML untuk akses kamera
st.write("### Deteksi Penyakit Tanaman Secara Real-time")
st.markdown(
    """
    <style>
    video {
        width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

html_code = """
    <video id="video" autoplay></video>
    <script>
    async function setupCamera() {
        const video = document.getElementById('video');
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // Capture frame setiap detik dan kirim ke Streamlit
        setInterval(async () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Kirim frame ke backend
            const dataUrl = canvas.toDataURL('image/jpeg');
            fetch('/upload_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: dataUrl })
            });
        }, 1000); // Interval 1 detik
    }

    setupCamera();
    </script>
"""

st.components.v1.html(html_code, height=300)

# Fungsi untuk melakukan prediksi pada frame
def predict(model, image):
    # Konversi frame ke tensor
    image = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image)
    return predictions[0]

# Fungsi untuk memproses data yang diterima dari frontend
def process_image(data_url):
    # Decode gambar dari format base64
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))
    predictions = predict(model, image)

    # Tambahkan kotak bounding dan label
    frame = np.array(image)
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= 0.8:
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(label.item(), "Unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert kembali ke format image yang bisa ditampilkan di Streamlit
    return Image.fromarray(frame)

# Tampilan prediksi
if 'uploaded_image' in st.session_state:
    frame = process_image(st.session_state['uploaded_image'])
    st.image(frame, caption="Hasil Deteksi", use_column_width=True)

