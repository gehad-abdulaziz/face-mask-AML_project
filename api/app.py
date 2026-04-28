"""
Face Mask Detection — FastAPI App
Role 6: API Developer — Gehad
"""

import io
import os
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
# استخدمنا Resize و CenterCrop عشان الصورة متمطش والموديل ميتلخبطش
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

# تحميل الموديل عند تشغيل السيرفر
model, CLASS_NAMES = load_mask_model(MODEL_PATH)

# ──────────────────────────────────────────
# FASTAPI APP SETUP
# ──────────────────────────────────────────
app = FastAPI(title="Face Mask Detection AI")

# التأكد من وجود فولدر static
if not os.path.exists("static"):
    os.makedirs("static")

# ربط ملفات الـ CSS/JS والـ HTML (اختياري لو هتحطي ملفات فرعية)
app.mount("/static", StaticFiles(directory="static"), name="static")

ACTION_MAP = {
    "WithMask"   : "Allow entry ✅",
    "WithoutMask": "Deny entry ❌ — Please wear a mask",
}

# ──────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────

# واجهة المستخدم الأساسية (الـ Dashboard الشيك)
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
        "model"    : "MobileNetV2"  # ضيفي السطر ده بالظبط
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