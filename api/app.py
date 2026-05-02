"""
Face Mask Detection — FastAPI App
Role 6: API Developer — Gehad
"""

import io
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
MODEL_PATH = "model/mask_detector.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────
# PREPROCESSING (Improved for accuracy)
# ──────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

# ──────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return m

def load_mask_model(path: str):
    checkpoint = torch.load(path, map_location=DEVICE)
    class_names = checkpoint.get("class_names", ["WithMask", "WithoutMask"])
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

model, CLASS_NAMES = load_mask_model(MODEL_PATH)

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ──────────────────────────────────────────
# FASTAPI APP SETUP
# ──────────────────────────────────────────
app = FastAPI(title="Face Mask Detection AI")

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

ACTION_MAP = {
    "WithMask"   : "Allow entry ✅",
    "WithoutMask": "Deny entry ❌ — Please wear a mask",
}

# ──────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """<h1>Dashboard UI not found!</h1><p>Please create <b>static/index.html</b></p>"""



@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "device"   : str(DEVICE),
        "classes"  : CLASS_NAMES,
        "model"    : "MobileNetV2" 
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    try:
        raw   = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error reading image: {e}")

    # Face Detection and Cropping
    try:
        # Convert PIL Image to OpenCV format
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Find the largest face by area
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Add a margin around the face (e.g., 20% to avoid cutting off masks)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.width, x + w + margin_x)
            y2 = min(image.height, y + h + margin_y)
            
            # Crop the face from the original image
            image = image.crop((x1, y1, x2, y2))
    except Exception as e:
        # If face detection fails for any unexpected reason, we fallback to original image
        print(f"Face detection fallback: {e}")

    # Inference
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    confidence  = float(probs.max())
    class_idx   = int(probs.argmax())
    class_label = CLASS_NAMES[class_idx]

    return {
        "status"    : "mask_on" if class_label == "WithMask" else "mask_off",
        "class"     : class_label,
        "action"    : ACTION_MAP.get(class_label, "Unknown"),
        "confidence": round(confidence, 4),
    }
