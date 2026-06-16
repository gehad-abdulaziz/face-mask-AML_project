"""
Face Mask Detection — FastAPI App
Role 6: API Developer — Gehad & Fatma
Updated: RetinaFace for better face detection (hijab + niqab + multi-face)
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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
MODEL_PATH = "model/mask_detector_v2.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])

# ──────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return m

def load_mask_model(path: str):
    checkpoint = torch.load(path, map_location=DEVICE)
    class_names = ["WithMask", "WithoutMask"]
    model = build_model(len(class_names))
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model, class_names

model, CLASS_NAMES = load_mask_model(MODEL_PATH)

# ──────────────────────────────────────────
# FACE DETECTOR — RetinaFace
# ──────────────────────────────────────────
try:
    from retinaface import RetinaFace
    USE_RETINAFACE = True
    print("✅ RetinaFace loaded")
except ImportError:
    USE_RETINAFACE = False
    print("⚠️  RetinaFace not found — falling back to Haarcascade")

# Haarcascade fallback
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces(image_pil: Image.Image):
    """
    Returns list of (x1, y1, x2, y2) face boxes.
    Uses RetinaFace if available, else Haarcascade.
    """
    img_np = np.array(image_pil)  # RGB
    faces  = []

    if USE_RETINAFACE:
        try:
            # RetinaFace يقبل RGB numpy array
            result = RetinaFace.detect_faces(img_np)
            if isinstance(result, dict):
                for face_data in result.values():
                    x1, y1, x2, y2 = face_data["facial_area"]
                    # margin 10%
                    w = x2 - x1
                    h = y2 - y1
                    x1 = max(0, x1 - int(w * 0.1))
                    y1 = max(0, y1 - int(h * 0.1))
                    x2 = min(image_pil.width,  x2 + int(w * 0.1))
                    y2 = min(image_pil.height, y2 + int(h * 0.1))
                    faces.append((int(x1), int(y1), int(x2), int(y2)))
        except Exception as e:
            print(f"RetinaFace error: {e}")

    if not faces:
        # Haarcascade fallback
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects   = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        for (x, y, w, h) in rects:
            faces.append((int(x), int(y), int(x + w), int(y + h)))

    return faces

# ──────────────────────────────────────────
# FASTAPI
# ──────────────────────────────────────────
app = FastAPI(title="Face Mask Detection AI")

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

ACTION_MAP = {
    "WithMask"   : "Allow entry ✅",
    "WithoutMask": "Deny entry ❌ — Please wear a mask",
}

def predict_face(face_pil: Image.Image):
    tensor = preprocess(face_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    idx   = int(probs.argmax())
    label = CLASS_NAMES[idx]
    conf  = round(float(probs[idx]), 4)
    return label, conf

# ──────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Dashboard UI not found!</h1>"

@app.get("/health")
def health():
    return {
        "status"       : "ok",
        "device"       : str(DEVICE),
        "classes"      : CLASS_NAMES,
        "model"        : "MobileNetV2",
        "face_detector": "RetinaFace" if USE_RETINAFACE else "Haarcascade",
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

    # ── Detect faces ──
    try:
        faces = detect_faces(image)
    except Exception as e:
        print(f"Detection error: {e}")
        faces = []

    # ── Inference ──
    if len(faces) == 0:
        per_face = []
        primary  = {
            "status"    : "mask_off",
            "class"     : "WithoutMask",
            "action"    : ACTION_MAP["WithoutMask"],
            "confidence": 0.0,
            "note"      : "No face detected",
        }
    else:
        per_face = []
        for i, (x1, y1, x2, y2) in enumerate(faces, start=1):
            face_crop   = image.crop((x1, y1, x2, y2))
            label, conf = predict_face(face_crop)
            per_face.append({
                "face_id"   : i,
                "bbox"      : {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                "status"    : "mask_on" if label == "WithMask" else "mask_off",
                "class"     : label,
                "action"    : ACTION_MAP.get(label, "Unknown"),
                "confidence": conf,
            })
        primary = max(per_face, key=lambda r: r["confidence"])

    return {
        "status"        : primary["status"],
        "class"         : primary["class"],
        "action"        : primary["action"],
        "confidence"    : primary["confidence"],
        "faces_detected": len(faces),
        "results"       : per_face,
    }