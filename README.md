# YOLO12 Object Detection Project (Webcam + Jupyter Notebook)

This project implements **real-time object detection** using the latest **Ultralytics YOLO12** model.  
The system is optimized to run inside **Jupyter Notebook**, avoiding all OpenCV window errors while providing smooth webcam output.

---

## 🚀 Features
- Real-time webcam detection using YOLO12 (latest Ultralytics model)
- Notebook-friendly display (no `cv2.imshow` required)
- Lightweight and fast inference using `yolo12n.pt`
- Compatible with Windows, Mac, and Linux
- Easy to modify for:
  - Video file detection
  - Saving screenshots
  - Model training on custom datasets

---

## 📁 Project Structure


project/
│
├── object_detection_notebook.ipynb # Main YOLO12 Notebook (safe for Jupyter)
├── yolo_webcam.py # Optional Python script for desktop mode
├── requirements.txt # All required dependencies
└── README.md # Documentation file


---

## 🧩 Installation

### 1️⃣ Install Python 3.10 or 3.11

### 2️⃣ Install PyTorch (recommended FIRST)
Visit: https://pytorch.org/get-started/locally/

Example CPU installation:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

3️⃣ Install project dependencies
pip install -r requirements.txt

📦 requirements.txt
ultralytics==8.2.0
opencv-python-headless==4.7.0.72
numpy
pandas
tqdm
Pillow
matplotlib


Using opencv-python-headless prevents GUI-related errors inside Jupyter Notebook.

▶️ Running YOLO12 in Jupyter Notebook

Open:

object_detection_notebook.ipynb


Run all cells.
The webcam output will appear inside the notebook, using IPython.display.

Example snippet used inside the notebook:

from ultralytics import YOLO
import cv2, time
from IPython.display import display, clear_output
from PIL import Image

model = YOLO("yolo12n.pt")
cap = cv2.VideoCapture(0)

start = time.time()
max_seconds = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()
    img = Image.fromarray(annotated)

    clear_output(wait=True)
    display(img)

    if time.time() > start + max_seconds:
        break

cap.release()
clear_output(wait=True)
print("Webcam released.")

🖥️ Running YOLO12 as a Normal Python Script (Desktop Mode)

If you want OpenCV windows on screen, use:

yolo_webcam.py
from ultralytics import YOLO
import cv2

model = YOLO("yolo12n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLO12 Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == 'q':
        break

cap.release()
cv2.destroyAllWindows()


Run it:

pip install opencv-python
python yolo_webcam.py

🛠️ Troubleshooting
❗ 1. cv2.imshow() not implemented

You're running inside a notebook → use notebook version
OR install GUI-OpenCV:

pip uninstall opencv-python-headless
pip install opencv-python

❗ 2. Webcam not opening

Try changing the index:

cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(2)

❗ 3. YOLO model not downloading

YOLO auto-downloads this file:

model = YOLO("yolo12n.pt")


Make sure you have internet.

📌 Notes

YOLO12 is the most recent stable YOLO model from Ultralytics.

yolo12n.pt is the nano version designed for fastest real-time performance.

You can train custom datasets easily:

model.train(data="data.yaml", epochs=50, imgsz=640)

✨ Author
dileepmedisetti

