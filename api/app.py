"""
Face Mask Detection — FastAPI App
Role 6: API Developer — Gehad & Fatma
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
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

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
# HELPER FUNCTIONS
# ──────────────────────────────────────────

def detect_faces_in_gray(gray_img, min_size=(80, 80)):
    """Run cascade detector on a grayscale image, return list of (x,y,w,h)."""
    return list(face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=15, # Increased from 8 to reduce false positives during movement
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    ))


def is_valid_face_rect(rect, img_w, img_h):
    """Filter out spurious face rectangles by size, shape, and image bounds."""
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return False

    aspect_ratio = w / h
    area = w * h
    image_area = img_w * img_h

    if area < image_area * 0.01:
        return False
    if area > image_area * 0.65:
        return False
    if h < img_h * 0.08:
        return False
    if y > img_h * 0.75:
        return False
    if aspect_ratio < 0.7 or aspect_ratio > 1.4:
        return False
    if x < -w * 0.1 or y < -h * 0.1:
        return False
    if x + w > img_w + w * 0.1 or y + h > img_h + h * 0.1:
        return False

    return True


def rotate_image(img_bgr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a BGR numpy image around its center by `angle` degrees."""
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def rotate_rect_back(
    x: int, y: int, w: int, h: int,
    angle: float, img_w: int, img_h: int
):
    """Map a rect found in a rotated image back to original-image coordinates."""
    cx_rot = x + w / 2
    cy_rot = y + h / 2
    M_inv = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), -angle, 1.0)
    pt = np.array([[[cx_rot, cy_rot]]], dtype=np.float32)
    pt_orig = cv2.transform(pt, M_inv)[0][0]
    x_orig = int(pt_orig[0] - w / 2)
    y_orig = int(pt_orig[1] - h / 2)
    return x_orig, y_orig, w, h


def iou(a, b) -> float:
    """Intersection-over-Union between two (x,y,w,h) rects."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms(rects: list, iou_thresh: float = 0.4) -> list:
    """Remove duplicate detections of the same face via Non-Maximum Suppression."""
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for r in rects:
        if all(iou(r, k) < iou_thresh for k in kept):
            kept.append(r)
    return kept


def predict_face(face_pil: Image.Image):
    """Run the mask classifier on a single cropped face PIL image."""
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
        return "<h1>Dashboard UI not found!</h1><p>Please create <b>static/index.html</b></p>"


@app.get("/health")
def health():
    return {
        "status" : "ok",
        "device" : str(DEVICE),
        "classes": CLASS_NAMES,
        "model"  : "MobileNetV2",
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

    # ── Face Detection across multiple rotation angles (Reduced to prevent false positives) ──
    ROTATION_ANGLES = [0, 15, -15]

    try:
        cv_img        = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_h, img_w  = cv_img.shape[:2]
        gray_original = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        min_face_size = (
            max(50, int(img_w * 0.06)),
            max(50, int(img_h * 0.06)),
        )

        all_rects = []
        for angle in ROTATION_ANGLES:
            gray = gray_original if angle == 0 else \
                   cv2.cvtColor(rotate_image(cv_img, angle), cv2.COLOR_BGR2GRAY)

            for rect in detect_faces_in_gray(gray, min_size=min_face_size):
                x, y, w, h = rect
                if angle != 0:
                    x, y, w, h = rotate_rect_back(x, y, w, h, angle, img_w, img_h)

                rect = (int(x), int(y), int(w), int(h))
                if is_valid_face_rect(rect, img_w, img_h):
                    all_rects.append(rect)

        faces = nms(all_rects, iou_thresh=0.5) if all_rects else []

    except Exception as e:
        print(f"Face detection error: {e}")
        faces = []

    # ── Per-face inference ──────────────────────────────────────────────
    if len(faces) == 0:
        per_face = []
        primary = {
            "status"    : "mask_off",
            "class"     : "WithoutMask",
            "action"    : ACTION_MAP["WithoutMask"],
            "confidence": 0.0,
            "note"      : "No face detected — please point the camera at a person",
        }
    else:
        per_face = []
        for i, (x, y, w, h) in enumerate(faces, start=1):
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x1 = max(0,            x - margin_x)
            y1 = max(0,            y - margin_y)
            x2 = min(image.width,  x + w + margin_x)
            y2 = min(image.height, y + h + margin_y)

            face_crop   = image.crop((x1, y1, x2, y2))
            label, conf = predict_face(face_crop)

            per_face.append({
                "face_id"   : i,
                "bbox"      : {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "status"    : "mask_on" if label == "WithMask" else "mask_off",
                "class"     : label,
                "action"    : ACTION_MAP.get(label, "Unknown"),
                "confidence": conf,
            })

        # ── Primary face = highest-confidence result (for backward compat) ──
        primary = max(per_face, key=lambda r: r["confidence"])

    return {
        # ── Backward-compatible top-level fields (test suite expects these) ──
        "status"        : primary["status"],
        "class"         : primary["class"],
        "action"        : primary["action"],
        "confidence"    : primary["confidence"],
        # ── Multi-face fields ──
        "faces_detected": len(faces),
        "results"       : per_face,
    }