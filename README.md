
# 🔍 Solar Panel Defect Detection

This project provides a computer vision solution to automatically detect and classify defects in solar panels using deep learning. It includes both **image classification** and **object detection** pipelines to help identify faults such as cracks, dirt, or physical damage.

---

## 📂 Project Structure

```
SolarPanel_Defect_Detection/
│
├── app.py                                     # (Optional) App launcher for frontend
│
├── classification/
│   └── classification_solar_panel.ipynb       # Notebook for image classification
│
├── Object detection/
│   └── object_detection.ipynb                 # Notebook for object detection
│
├── models/
│   ├── classification_model.pt                # Trained classification model (PyTorch)
│   └── object_detection_model.pt              # Trained object detection model (PyTorch)
```

---

## 🧠 Model Downloads

- **🔗 Classification Model (.pt):**  
  [Download here](https://drive.google.com/file/d/1T_gGiShiKLGOYHnvHGQCSuNAX7bL48qo/view?usp=sharing)

- **🔗 Object Detection Model (.pt):**  
  [Download here](https://drive.google.com/file/d/1pAu72yp_zaxMmtF1poAFX6x6oglPSB7l/view?usp=sharing)

Place the downloaded models in a new folder named `models/` in the project root.

---

## 🗂️ Dataset Links

- **📦 Classification Dataset:**  
  [Download here](https://drive.google.com/file/d/1tz5DGh3N4MHcJxtb_Be6M6uqsURVyk7s/view?usp=sharing)

- **📦 Object Detection Dataset:**  
  [Download here](https://drive.google.com/file/d/1ejQdoYZGT6T0RTt2BBYLJw_ld-TeO5Jl/view?usp=sharing)

After downloading:
- Extract the datasets.
- Update paths in notebooks if needed.

---

## ⚙️ Installation & Requirements

Ensure Python 3.7+ is installed.

Install dependencies using:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install torch torchvision opencv-python matplotlib scikit-learn jupyter notebook
```

---

## 🚀 How to Run

### 1. **Classification Pipeline**

```bash
cd classification
jupyter notebook classification_solar_panel.ipynb
```

### 2. **Object Detection Pipeline**

```bash
cd "Object detection"
jupyter notebook object_detection.ipynb
```

### 3. **(Optional) Run Frontend App**

If `app.py` uses Streamlit:

```bash
streamlit run app.py
```

---

## 📊 Features

- **Defect Classification:** Identifies the condition of a solar panel image (e.g., normal, cracked).
- **Defect Detection:** Locates and highlights defects using bounding boxes.

---

## 📌 Notes

- Models are pre-trained and can be used for inference immediately.
- Dataset paths may need editing in the notebooks after extraction.
- Results include prediction visualizations and evaluation metrics (e.g., accuracy, IoU).

---

## 👩‍💻 Author

**R. R. Shanmugapriya**  
📧 rrshanmugapriya01@gmail.com

---

## 📄 License

MIT License *(or your applicable license)*
