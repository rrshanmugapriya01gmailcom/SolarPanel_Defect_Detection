# --- Full Updated Streamlit Code with Simplified "Why Choose FaultyLens" Section ---

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import os

# --- Streamlit Config ---
st.set_page_config(
    page_title="Faulty Lens: The eye that never misses",
    layout="wide"
)

# --- Custom CSS ---
custom_css = """
<style>
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #4d5654;
        color: black;
    }
    .hero-container {
        position: relative;
        height: 400px;
        overflow: hidden;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    .hero-video {
        position: absolute;
        top: 50%;
        left: 50%;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: 0;
        transform: translate(-50%, -50%);
        opacity: 0.4;
    }
    .hero-text {
        position: relative;
        z-index: 1;
        text-align: center;
        color: white;
        padding-top: 6rem;
    }
    .hero-text h1 {
        font-size: 4.5rem;
        font-weight: bold;
        color: #003366;
    }
    .hero-text h3 {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    .why-faultylens-section {
        background: #1d3b23;
        padding: 4rem 1rem;
        color: white;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    .feature-item {
        background-color: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: white;
    }
    .feature-item h3 {
        font-size: 1.2rem;
        color: #A5D6A7;
        margin-bottom: 0.5rem;
    }
    .classification-result {
        background-color: #a0522d;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .reduced-img {
        max-width: 400px;
        margin: 1rem auto;
        display: block;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Hero Section with Video Background ---
st.markdown("""
<h3 style='color:white; text-align:center;'>🎥 Live Demo: FaultyLens in Action</h3>
<iframe width="720" height="405" src="https://www.youtube.com/embed/FNSACEFzkMY" 
frameborder="0" allowfullscreen style="display: block; margin: auto; border-radius: 10px;"></iframe>
""", unsafe_allow_html=True)



# --- "Why FaultyLens?" Section ---
st.markdown('<a name="why-faultylens"></a>', unsafe_allow_html=True)
st.markdown(
    """
    <section class="why-faultylens-section">
        <div class="container">
            <h2 class="section-heading">Why Choose FaultyLens?</h2>
            <p class="text-center text-xl mb-8 max-w-3xl mx-auto">
                Our advanced AI system ensures your solar assets perform at their peak, minimizing downtime and maximizing energy output.
            </p>
            <div class="feature-grid">
                <div class="row" style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem; margin-bottom: 2rem;">
                    <div class="feature-item">
                        <h3>🔍 Object Detection</h3>
                        <p>Precisely locate and identify specific types of physical damage and anomalies on solar panels, such as cracks, hot spots, and module discoloration.</p>
                    </div>
                    <div class="feature-item">
                        <h3>📊 Image Classification</h3>
                        <p>Categorize detected issues like dust, bird droppings, electrical damage, or snow cover, enabling targeted and efficient maintenance strategies.</p>
                    </div>
                    <div class="feature-item">
                        <h3>⚡ Enhanced Efficiency</h3>
                        <p>Automate inspections, drastically reducing manual labor, time, and operational costs for large-scale solar farms.</p>
                    </div>
                </div>
                <div class="row" style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;">
                    <div class="feature-item">
                        <h3>🎯 High Accuracy</h3>
                        <p>Leverage robust AI models trained on diverse datasets for unparalleled precision in fault identification and analysis.</p>
                    </div>
                    <div class="feature-item">
                        <h3>📈 Optimized Performance</h3>
                        <p>Proactive detection and resolution of faults lead to consistent peak performance and extended lifespan of solar assets.</p>
                    </div>
                    <div class="feature-item">
                        <h3>🌐 Scalability</h3>
                        <p>Our solution is designed to be easily deployed and scaled across various solar installation sizes, from small arrays to vast solar parks.</p>
                    </div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)

# --- Load Models ---
@st.cache_resource
def load_classification_model():
    model = torch.load("best_resnet152_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

@st.cache_resource
def load_detection_model():
    return YOLO("best_yolo11.pt")

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# --- Upload Section ---
st.header(" Upload Solar Panel Image for Analysis")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=False, output_format="auto", width=400)

    cls_model = load_classification_model()
    det_model = load_detection_model()

    st.subheader("📌 Classification Result")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = cls_model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        class_name = class_labels[predicted_idx] if predicted_idx < len(class_labels) else "Unknown"
        st.markdown(f"<div class='classification-result'>🔎 Predicted Class: <strong>{class_name.upper()}</strong></div>", unsafe_allow_html=True)

    st.subheader("📌 Object Detection Result (YOLO)")
    temp_path = "temp.jpg"
    image.save(temp_path)
    results = det_model(temp_path, conf=0.6)
    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        st.warning("⚠️ No objects detected.")
    else:
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Detected Objects", use_container_width=False, channels="BGR", width=500)

        st.subheader("🔍 Detected Details")
        for box in detections:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"🟢 Class {class_id} - Confidence: {conf:.2f}")

    os.remove(temp_path)
else:
    st.info("📅 Upload a solar panel image to begin analysis.")
