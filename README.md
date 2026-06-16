# 😷 Face Mask Detection — AML Project

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=for-the-badge&logo=pytorch) 
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi) 
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?style=for-the-badge&logo=opencv)

> A robust, end-to-end deep learning system that detects whether a person is wearing a face covering or not. Powered by **MobileNetV2** for classification and **RetinaFace** for highly accurate multi-face detection. Deployed as a microservice architecture using **FastAPI**, **Streamlit**, and **Docker**.

---

## ✨ Key Features & Breakthroughs

- **Diverse Cultural Recognition:** Model augmented with **Hijab and Niqab** images to ensure high accuracy across diverse demographics.
- **Multi-Face "Gang" Detection:** Integrated **RetinaFace** to instantly detect and classify multiple people in crowded scenes simultaneously.
- **Full-Body Handling:** Preprocessing pipeline automatically hunts for faces within full-body images, ensuring the classifier isn't confused by background noise.
- **High Performance:** Achieved **99.90% Accuracy** via a tailored two-phase fine-tuning strategy.

---

## 📊 Dataset & Real-World Adaptations

- **Base Source:** [Face Mask 12K Images Dataset — Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- **Classes:** `WithMask` / `WithoutMask`
- **Total Base Images:** ~12,000 real-world face photos

### 🛠️ Dataset Challenges & Solutions

1. **Lack of Diversity:** Realized the dataset lacked representation of women wearing Hijabs and Niqabs. **Solution:** Manually sourced and injected these images to prevent bias.
2. **Full-Body Image Failure:** The model was trained on cropped faces and failed on full-body images. **Solution:** Implemented **RetinaFace** preprocessing to crop faces dynamically before classification.

| Split | WithMask | WithoutMask | Total |
|-------|----------|-------------|-------|
| Train | 4,982 | 4,983 | 9,965 |
| Validation | 398 | 400 | 798 |
| Test | 482 | 506 | 988 |

---

## 🏗️ Architecture & Model Pipeline

### 1. Face Detection Pipeline (RetinaFace)
Before classification, every image passes through **RetinaFace** (with Haarcascade fallback). It scans the image, extracts bounding boxes for all faces (even at extreme angles or partial occlusions), and crops them.

### 2. Classification Model (MobileNetV2)
- **Architecture:** MobileNetV2 (Transfer Learning — pretrained on ImageNet)
- **Framework:** PyTorch + TorchVision
- **Classifier Head:** `Linear(1280 → 256) → ReLU → Dropout(0.2) → Linear(256 → 2)`
- **Model File Size:** ~13.6 MB

### Training Strategy
Training was split into two phases:
1. **Phase 1 (Head Training):** Base layers frozen. Custom classifier head trained for 10 epochs.
2. **Phase 2 (Fine-tuning):** Top layers unfrozen. Full model fine-tuned with a lower learning rate (early stopped at epoch 13).

### Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | **99.90%** ✅ |
| Precision | 0.9990 |
| Recall | 0.9990 |
| F1-Score | 0.9990 |

> Target accuracy was > 90% — **achieved with a significant margin.**

---

## 🎨 Data Augmentation Strategy

Designed a **data-driven augmentation pipeline** based on EDA insights to handle real-world camera imperfections (blur, low light, bad angles).

| Challenge | Augmentation Applied |
|-----------|---------------------|
| Blur (~20% of data) | `GaussianBlur (p=0.1)` |
| Dark images | `ColorJitter (brightness=0.2)` |
| Overexposed images | `ColorJitter (contrast=0.2)` |
| Camera angle variation | `RandomPerspective (p=0.2)` |
| Face orientation | `RandomHorizontalFlip (p=0.5)` |
| Head tilt | `RandomRotation (15)` |

---

## 🚀 Deployment (Docker & API)

The project is fully containerized into two distinct microservices:

| Container | Port | Service |
|-----------|------|---------|
| `api` | 8000 | **FastAPI** — AI Inference & Prediction |
| `streamlit` | 8501 | **Streamlit** — Web UI Dashboard |

### Running the Stack via Docker
```bash
# Build and run all services
docker compose -f docker/docker-compose.yml up --build
```
*Access the UI at `http://localhost:8501` and the API Docs at `http://localhost:8000/docs`.*

### Running Locally (Without Docker)
```bash
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

---

## 📸 Test Results in Action

### "Gang" Test (Multi-Face Detection via RetinaFace)
<img width="1228" height="672" alt="image" src="https://github.com/user-attachments/assets/a7152e30-97b2-4df6-8e27-474c5ed4e7ce" />
<img width="1235" height="613" alt="image" src="https://github.com/user-attachments/assets/bf33ce8a-80c3-48f8-b226-41c2abf3a94c" />


---

## 📁 Project Structure

```
face-mask-AML_project/
├── api/                  # FastAPI app (app.py) & testing scripts
├── data/                 # Raw/Split dataset (Ignored in Git)
├── docker/               # Dockerfiles and docker-compose.yml
├── model/                # Trained model weights (.pth)
├── notebooks/            # EDA & Evaluation Jupyter Notebooks
├── presentation/         # Demo slides and assets
├── reports/              # Metrics, charts, and augmentation visuals
├── src/                  # PyTorch training, loading, and augmentation scripts
├── ui/                   # Streamlit Frontend application
└── requirements.txt      # Python dependencies
```

---

## 🛠️ How to Contribute

1. Clone the repo
2. Create your branch: `git checkout -b feature/your-feature-name`
3. Commit your work: `git commit -m "add: description of the feature"`
4. Push: `git push origin feature/your-feature-name`
5. Open a Pull Request targeting `main`

