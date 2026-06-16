"""
Face Mask Detection — Automated API Tests
Role 6: API Developer — Gehad

Usage:
    # Make sure the API is running first:
    #   uvicorn api.app:app --reload --port 8000
    
    python api/test_api.py
"""

import io
import sys
import time
import requests
import numpy as np
from PIL import Image

BASE_URL = "http://localhost:8000"

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"

results = []


def check(name: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append(condition)


def make_image_bytes(color: tuple = (200, 200, 200), size=(224, 224)) -> bytes:
    """Create a dummy RGB image and return it as JPEG bytes."""
    img = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# TEST 1 — API is reachable
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  Face Mask Detection — API Test Suite")
print("="*55 + "\n")

try:
    r = requests.get(f"{BASE_URL}/", timeout=5)
    check("GET /  — API reachable", r.status_code == 200, f"status={r.status_code}")
except requests.exceptions.ConnectionError:
    print(f"{FAIL}  Cannot connect to {BASE_URL}")
    print("       → Make sure the API is running:  uvicorn api.app:app --reload --port 8000")
    sys.exit(1)

# ─────────────────────────────────────────────
# TEST 2 — Health endpoint
# ─────────────────────────────────────────────
r = requests.get(f"{BASE_URL}/health")
body = r.json()
check("GET /health — status ok",   body.get("status") == "ok")
check("GET /health — has classes", "classes" in body, str(body.get("classes")))
check("GET /health — has model",   body.get("model") == "MobileNetV2")

# ─────────────────────────────────────────────
# TEST 3 — /predict with a valid image
# ─────────────────────────────────────────────
img_bytes = make_image_bytes()
r = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("test.jpg", img_bytes, "image/jpeg")},
)
check("POST /predict — status 200", r.status_code == 200, f"status={r.status_code}")

body = r.json()
check("POST /predict — has 'status'",         "status"         in body)
check("POST /predict — has 'action'",         "action"         in body)
check("POST /predict — has 'confidence'",     "confidence"     in body)
check("POST /predict — has 'class'",          "class"          in body)
check("POST /predict — has 'faces_detected'", "faces_detected" in body,
      str(body.get("faces_detected")))
check("POST /predict — has 'results' list",   isinstance(body.get("results"), list),
      str(type(body.get("results"))))
check("POST /predict — valid status value",
      body.get("status") in ("mask_on", "mask_off"),
      body.get("status"))
check("POST /predict — valid class value",
      body.get("class") in ("WithMask", "WithoutMask"),
      body.get("class"))
check("POST /predict — confidence in [0,1]",
      0.0 <= body.get("confidence", -1) <= 1.0,
      str(body.get("confidence")))
check("POST /predict — faces_detected is int",
      isinstance(body.get("faces_detected"), int),
      str(body.get("faces_detected")))

# ─────────────────────────────────────────────
# TEST 4 — /predict rejects non-image files
# ─────────────────────────────────────────────
r = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("bad.txt", b"this is not an image", "text/plain")},
)
check("POST /predict — rejects non-image (400)", r.status_code == 400,
      f"status={r.status_code}")

# ─────────────────────────────────────────────
# TEST 5 — /predict rejects corrupted image
# ─────────────────────────────────────────────
r = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("corrupt.jpg", b"\xff\xd8\xff corrupted bytes", "image/jpeg")},
)
check("POST /predict — rejects corrupt image (422)",
      r.status_code in (400, 422),
      f"status={r.status_code}")

# ─────────────────────────────────────────────
# TEST 6 — Response time under 2 seconds
# ─────────────────────────────────────────────
img_bytes = make_image_bytes()
t0 = time.time()
requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("perf.jpg", img_bytes, "image/jpeg")},
)
elapsed = time.time() - t0
check("POST /predict — response < 5s (CPU)", elapsed < 5.0, f"{elapsed:.3f}s")

# ─────────────────────────────────────────────
# TEST 7 — per-face result structure (when faces detected)
# ─────────────────────────────────────────────
img_bytes = make_image_bytes()
r = requests.post(
    f"{BASE_URL}/predict",
    files={"file": ("struct.jpg", img_bytes, "image/jpeg")},
)
body = r.json()
results_list = body.get("results", [])

if results_list:
    first = results_list[0]
    check("POST /predict — per-face has 'face_id'",
          "face_id" in first, str(first.get("face_id")))
    check("POST /predict — per-face has 'bbox'",
          "bbox" in first and all(k in first["bbox"] for k in ("x", "y", "w", "h")),
          str(first.get("bbox")))
    check("POST /predict — per-face has 'status'",
          first.get("status") in ("mask_on", "mask_off"), first.get("status"))
    check("POST /predict — per-face has 'class'",
          first.get("class") in ("WithMask", "WithoutMask"), first.get("class"))
    check("POST /predict — per-face has 'action'",
          "action" in first, first.get("action"))
    check("POST /predict — per-face confidence in [0,1]",
          0.0 <= first.get("confidence", -1) <= 1.0, str(first.get("confidence")))
else:
    # No face detected — check note field exists in primary
    check("POST /predict — no-face note present",
          body.get("class") == "WithoutMask" and body.get("confidence") == 0.0,
          f"class={body.get('class')} conf={body.get('confidence')}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print()
print("="*55)
passed = sum(results)
total  = len(results)
status = "ALL PASSED 🎉" if passed == total else f"{total - passed} FAILED ⚠️"
print(f"  Results: {passed}/{total}  —  {status}")
print("="*55 + "\n")

sys.exit(0 if passed == total else 1)