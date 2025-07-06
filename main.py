from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import io
import base64
import numpy as np
import cv2

# --- FastAPI Setup ---
app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models ---
classification_model = torch.load("best_resnet152_model.pt", map_location=torch.device("cpu"))
classification_model.eval()
detection_model = YOLO("/home/dharun/Desktop/solar_panel/best_yolo11.pt")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

def pil_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # --- Classification ---
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        class_name = class_labels[predicted_idx] if predicted_idx < len(class_labels) else "Unknown"

    # --- Detection ---
    temp_path = "temp.jpg"
    image.save(temp_path)
    results = detection_model(temp_path, conf=0.6)
    detections = results[0].boxes

    boxes = []
    if detections:
        for box in detections:
            boxes.append({
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0])
            })
        detected_image = results[0].plot()
        _, img_encoded = cv2.imencode('.jpg', detected_image)
        b64_encoded = base64.b64encode(img_encoded).decode()
    else:
        b64_encoded = pil_to_base64(image)

    return JSONResponse(content={
        "classification": class_name,
        "detections": boxes,
        "detection_image": f"data:image/jpeg;base64,{b64_encoded}"
    })

@app.get("/")
def read_root():
    return {"message": "Welcome to the Faulty Lens API. Use /analyze to POST images."}
