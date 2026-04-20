# Face Mask Detection — AML Project

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

> A deep learning system that detects whether a person is wearing a face mask or not, built with MobileNetV2 transfer learning and deployed via FastAPI + Docker.

---

## Team

| # | Role | Member |
|---|------|--------|
| 1 | Data Manager | Jumana |
| 2 | EDA & Visualizer | Adam |
| 3 | Augmentation Designer | Ahmed |
| 4 | Model Trainer | Ghada |
| 5 | Evaluator | Doha |
| 6 | API Developer | Gehad |
| 7 | Deployer & Presenter | Fatima |

---

## Dataset

- **Source:** [Face Mask 12K Images Dataset — Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- **Classes:** `WithMask` / `WithoutMask`
- **Total Images:** ~12,000 real-world face photos
- **Split:** Train / Validation / Test (pre-organized)

---

## Project Structure

```
face-mask-AML_project/
│
├── data/                        # Dataset folder (not uploaded to GitHub)
│   ├── Train/
│   │   ├── WithMask/
│   │   └── WithoutMask/
│   ├── Validation/
│   └── Test/
│
├── notebooks/
│   ├── eda_notebook.ipynb       # EDA + 8 visualizations (Adam)
│   └── evaluation.ipynb        # Confusion matrix + metrics (Doha)
│
├── src/
│   ├── data_loader.py           # PyTorch DataLoader setup (Jumana)
│   ├── verify_dataset.py        # Check for corrupt images (Jumana)
│   ├── augmentation.py          # Augmentation pipeline (Ahmed)
│   └── train.py                 # Training script — MobileNetV2 (Ghada)
│
├── model/
│   └── mask_detector.pth        # Trained model weights (Ghada)
│
├── api/
│   ├── app.py                   # FastAPI app — /predict endpoint (Gehad)
│   └── test_api.py              # Automated API tests (Gehad)
│
├── docker/
│   ├── Dockerfile               # Container setup (Fatima)
│   └── docker-compose.yml       # Full stack deployment (Fatima)
│
├── reports/
│   ├── dataset_report.md        # Class balance + data stats (Jumana)
│   ├── metrics_report.md        # F1, Precision, Recall, Accuracy (Doha)
│   └── training_curves.png      # Loss/Accuracy plots (Ghada)
│
├── presentation/
│   └── demo_slides.pdf          # 10-minute presentation (Fatima)
│
├── .gitignore
└── README.md
```

---

## Model

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Framework:** PyTorch + TorchVision
- **Target Accuracy:** > 90%
- **Export Format:** `.pth`

---

## API

```bash
# Run locally
uvicorn api.app:app --reload --port 8000

# Swagger docs
http://localhost:8000/docs
```

**Example response:**
```json
{
  "status": "mask_on",
  "action": "Allow entry",
  "confidence": 0.96
}
```

---

## Docker

```bash
# Build and run
docker-compose up --build

# Test the API
python api/test_api.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch + TorchVision | Model training & transfer learning |
| FastAPI + Uvicorn | REST API deployment |
| Docker | Containerized deployment |
| Pillow | Image preprocessing |
| scikit-learn | Metrics & evaluation |

---

## Timeline

| Week | Milestone | Members |
|------|-----------|---------|
| 1 | Dataset setup, EDA, Augmentation pipeline | Jumana, Adam, Ahmed |
| 2 | Model training, evaluation, export .pth | Ghada, Doha |
| 3 | FastAPI app, Docker, test script | Gehad, Fatima |
| 4 | Integration test, presentation, live demo | All |

---

## Installation

```bash
pip install torch torchvision fastapi uvicorn python-multipart Pillow scikit-learn
```

---

## How to Contribute

1. Clone the repo
2. Create your branch: `git checkout -b feature/your-name`
3. Commit your work: `git commit -m "add: your task description"`
4. Push: `git push origin feature/your-name`
5. Open a Pull Request to `main`
