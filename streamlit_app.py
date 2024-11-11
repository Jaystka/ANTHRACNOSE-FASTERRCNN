import streamlit as st
import av
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2

# Daftar nama kelas untuk deteksi
CLASS_NAMES = {1: "healthy", 2: "anthracnose"}

# Fungsi untuk memuat model Faster R-CNN
@st.cache_resource
def load_model():
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=3)
    # Pastikan untuk mengisi path model yang benar
    model.load_state_dict(torch.load("fasterrcnn_anthracnose_detector13.pth", map_location=torch.device('cpu'), , weights_only=True))
    model.eval()
    return model

model = load_model()

class VideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model

    def recv(self, frame):
        # Konversi frame ke RGB
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        # Prediksi
        with torch.no_grad():
            predictions = self.model([F.to_tensor(pil_image)])[0]

        # Tampilkan bounding box dan label pada frame
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score >= 0.8:  # Threshold
                x1, y1, x2, y2 = map(int, box)
                class_name = CLASS_NAMES.get(label.item(), "Unknown")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Menjalankan webrtc streamer
webrtc_streamer(key="example", video_processor_factory=lambda: VideoProcessor(model))
