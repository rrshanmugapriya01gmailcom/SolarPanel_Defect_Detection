# --- Streamlit Frontend Calling FastAPI Backend ---

import streamlit as st
from PIL import Image
import requests
import io

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
    .hero-background {
        background-image: url('https://images.unsplash.com/photo-1558449028-b53a39d100fc?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'); /* Replace with your image URL */
        background-size: cover;
        background-position: center;
        padding: 4rem 1rem;
        text-align: center;
        border-radius: 1rem;
    }
    .hero-text h1 {
        font-size: 6rem;
        font-weight: bold;
        color: black;
        text-transform: uppercase;
    }
    .hero-text h3 {
        font-size: 2rem;
        font-weight: bold;
        color: black;
        text-transform: uppercase;
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
        grid-template-columns: repeat(2, 1fr);
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
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Hero Section with Background Image ---
st.markdown("""
<div class="hero-background">
    <div class="hero-text">
        <h1>Faulty Lens</h1>
        <h3>The eye that never misses</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# --- "Why FaultyLens?" Section ---
st.markdown("""
<section class="why-faultylens-section">
    <div class="container">
        <h2 class="section-heading">Why Choose FaultyLens?</h2>
        <p class="text-center text-xl mb-8 max-w-3xl mx-auto">
            Our advanced AI system ensures your solar assets perform at their peak, minimizing downtime and maximizing energy output.
        </p>
        <div class="feature-grid">
            <div class="feature-item">
                <h3>🔍 Object Detection</h3>
                <p>Detect cracks, hotspots, and discoloration.</p>
            </div>
            <div class="feature-item">
                <h3>📊 Image Classification</h3>
                <p>Classify issues like dust, droppings, damage, or snow cover.</p>
            </div>
            <div class="feature-item">
                <h3>⚡ Enhanced Efficiency</h3>
                <p>Reduce manual inspection costs and time.</p>
            </div>
            <div class="feature-item">
                <h3>🎯 High Accuracy</h3>
                <p>Trained on diverse datasets for top precision.</p>
            </div>
            <div class="feature-item">
                <h3>📈 Optimized Performance</h3>
                <p>Ensure longer solar panel lifespan with proactive care.</p>
            </div>
            <div class="feature-item">
                <h3>🌐 Scalability</h3>
                <p>Deploy across any size solar installation.</p>
            </div>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

# --- Upload Section ---
st.header(" Upload Solar Panel Image for Analysis")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    with st.spinner("Classifying and detecting..."):
        response = requests.post(
            "http://127.0.0.1:8000/analyze",
            files={"file": ("image.jpg", img_bytes, "image/jpeg")}
        )

        

    if response.status_code == 200:
        data = response.json()
        st.subheader("📌 Classification Result")
        st.markdown(f"<div class='classification-result'>🔎 Predicted Class: <strong>{data['classification'].upper()}</strong></div>", unsafe_allow_html=True)

        st.subheader("📌 Object Detection")
        st.image(data["detection_image"], caption="Detected Objects", width=500)

        st.subheader("🔍 Detected Boxes")
        for i, det in enumerate(data["detections"], 1):
            st.write(f"🟢 Class {det['class_id']} - Confidence: {det['confidence']:.2f}")
    else:
        st.error("❌ Error occurred while processing the image.")
else:
    st.info("📅 Upload a solar panel image to begin analysis.")
