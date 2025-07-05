@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print("📥 File received for processing")
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Classification
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = classification_model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            class_name = class_labels[predicted_idx] if predicted_idx < len(class_labels) else "Unknown"

        # Detection
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
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
