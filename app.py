import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import os

# Load classification model
@st.cache_resource
def load_classification_model():
    model = torch.load("best_resnet152_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

# Load YOLOv8 object detection model
@st.cache_resource
def load_detection_model():
    return YOLO("/home/dharun/Desktop/solar_panel/solar_panel_object_detection/runs/detect/train2/weights/best.pt")

# Preprocessing for classification model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels for classification
class_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# App Title
st.title("🔍 Solar Panel Fault Detection (Classification + Object Detection)")

# File upload
uploaded_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Load models
    cls_model = load_classification_model()
    det_model = load_detection_model()

    # -------------------- Classification --------------------
    st.subheader("📌 Classification Result")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = cls_model(input_tensor)

        if output.ndim == 2 and output.shape[0] == 1:
            predicted_idx = torch.argmax(output, dim=1).item()
        else:
            st.error("Unexpected output shape from classification model!")
            predicted_idx = -1

        if 0 <= predicted_idx < len(class_labels):
            class_name = class_labels[predicted_idx]
        else:
            class_name = "Unknown"

        st.success(f"🔎 Predicted Class: **{class_name.upper()}**")

    # -------------------- Object Detection --------------------
    st.subheader("📌 Object Detection Result (YOLO)")
    with st.spinner("Running detection..."):
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        # Run YOLO with confidence threshold
        results = det_model(temp_path, conf=0.6)
        detections = results[0].boxes  # YOLO detections

        if detections is None or len(detections) == 0:
            st.warning("⚠️ No objects detected in the image.")
            st.image(image, caption="Original Image - No Detections", use_container_width=True)
        else:
            # Show image with bounding boxes
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Detection Output with Bounding Boxes", use_container_width=True, channels="BGR")

            # Show labels and confidence scores
            st.subheader("🔍 Detected Objects")
            names = det_model.names

            for box in detections:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= 0.6:  # Ensure threshold applied again
                    label = names[class_id] if names and class_id in names else f"Class {class_id}"
                    st.write(f"🟩 **{label}** — Confidence: {conf:.2f}")

        # Cleanup temp image
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.info("📥 Please upload an image file to run detection and classification.")
