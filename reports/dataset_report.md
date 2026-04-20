# Dataset Report – Face Mask Detection

## 1. Introduction

This report presents the preprocessing, validation, and quality assessment of the dataset used for the Face Mask Detection project. The goal is to ensure the dataset is clean, balanced, and suitable for training a deep learning model (MobileNetV2).

---

## 2. Dataset Overview

The dataset consists of real-world facial images categorized into two classes:

* **WithMask**
* **WithoutMask**

The images include variations in:

* lighting conditions (indoor and outdoor)
* face orientations and angles
* mask types (surgical, cloth, etc.)
* backgrounds and occlusions

---

## 3. Dataset Structure

```plaintext
data/
├── Train/
├── Validation/
└── Test/
```

Each split contains:

```plaintext
WithMask/
WithoutMask/
```

---

## 4. Dataset Statistics

## 4.1 Before Cleaning

### Training Set

* WithMask: **5000**
* WithoutMask: **5000**
* Total: **10000**

### Validation Set

* WithMask: **400**
* WithoutMask: **400**
* Total: **800**

### Test Set

* WithMask: **483**
* WithoutMask: **509**
* Total: **992**

---

## 4.2 After Cleaning

### Training Set

* WithMask: **4982**
* WithoutMask: **4983**
* Total: **9965**

### Validation Set

* WithMask: **398**
* WithoutMask: **400**
* Total: **798**

### Test Set

* WithMask: **482**
* WithoutMask: **506**
* Total: **988**

---

##  4.3 Cleaning Impact

| Split      | Before    | After     | Removed |
| ---------- | --------- | --------- | ------- |
| Train      | 10000     | 9965      | 35      |
| Validation | 800       | 798       | 2       |
| Test       | 992       | 988       | 4       |
| **Total**  | **11792** | **11751** | **41**  |

### Interpretation:

* Only **41 images (~0.35%)** were removed
* Cleaning improved quality **without affecting dataset size significantly**

---

## 5. Data Cleaning Process

### Manual Cleaning

The following images were removed:

*  Completely black images
*  Images without visible faces (e.g., hair only, ears only)
*  Incorrectly labeled samples (e.g., no mask in *WithMask*)

### Strategy:

* Kept slightly blurred images (to preserve realism)
* Kept lighting variations (important for generalization)

---

## 6. Data Quality Analysis

Automated verification results:

* **Corrupted images:** 0 (0%) 
* **Blurry images:** 59 (~0.5%) 
* **Dark images:** 48 (~0.4%) 
* **Overexposed images:** 117 (~1%) 

### Interpretation:

* Blur level is **very low**
* Lighting variation is **healthy and realistic**
* Dataset maintains **diversity for better model generalization**

---

## 7. Class Balance Analysis

| Split      | WithMask | WithoutMask | Balance    |
| ---------- | -------- | ----------- | ---------- |
| Train      | 4982     | 4983        | Excellent  |
| Validation | 398      | 400         | Excellent  |
| Test       | 482      | 506         | Acceptable |

### Conclusion:

The dataset remains **well-balanced**, ensuring no bias during training.

---

## 8. Data Verification Pipeline

A verification pipeline was implemented to ensure dataset integrity:

*  Folder structure validation
*  Corruption detection
*  Blur detection (Laplacian variance)
*  Brightness analysis
*  Class distribution validation

---

## 9. Data Loading Pipeline

A PyTorch DataLoader pipeline was implemented using:

* `torchvision.datasets.ImageFolder`
* Image resizing to **224×224**
* Tensor conversion and normalization
* Efficient batching and shuffling

All loaders were successfully tested:

* Correct image shapes: **[32, 3, 224, 224]**
* Correct label mapping and batching

---

## 10. Final Evaluation

After preprocessing, the dataset is:

*  Clean and free from corrupted data
*  Balanced across classes
*  Representative of real-world conditions
*  Suitable for deep learning training

---

## Conclusion

The dataset is **fully prepared and ready** for training a MobileNetV2-based face mask detection model. The cleaning process improved data quality while preserving diversity, ensuring strong generalization and expected high performance.

---

