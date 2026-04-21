import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_PATH = "../data"


# 1) Structure Check

def check_structure():
    print("\n Checking folder structure...\n")

    splits = ["Train", "Validation", "Test"]
    classes = ["WithMask", "WithoutMask"]

    ok = True

    for split in splits:
        for cls in classes:
            path = os.path.join(DATA_PATH, split, cls)
            if os.path.exists(path):
                print(f" OK: {path}")
            else:
                print(f" Missing: {path}")
                ok = False

    return ok



# 2) Corrupted Images

def check_corrupted():
    print("\n Checking corrupted images...\n")

    corrupted = []

    files = []
    for root, _, f in os.walk(DATA_PATH):
        for file in f:
            files.append(os.path.join(root, file))

    for path in tqdm(files, desc="Scanning"):
        try:
            img = Image.open(path)
            img.verify()
        except:
            corrupted.append(path)

    print(f" Corrupted images found: {len(corrupted)}")
    return corrupted


# 3) Blur Detection

def check_blur(threshold=50):
    print("\n Checking blur level...\n")

    blurry = []
    sharp_scores = []

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            path = os.path.join(root, file)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            score = cv2.Laplacian(img, cv2.CV_64F).var()
            sharp_scores.append(score)

            if score < threshold:
                blurry.append(path)

    print(f" Blurry images: {len(blurry)}")
    print(f" Avg sharpness: {np.mean(sharp_scores):.2f}")

    return blurry



# 4) Brightness Check

def check_brightness():
    print("\n Checking brightness...\n")

    dark, bright = 0, 0
    total = 0

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            path = os.path.join(root, file)

            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)

            total += 1

            if mean < 50:
                dark += 1
            elif mean > 200:
                bright += 1

    print(f" Dark images: {dark}")
    print(f" Overexposed images: {bright}")

    return dark, bright



# 5) Dataset Counts per Folder

def check_balance():
    print("\n Dataset Counts per Folder...\n")

    for split in ["Train", "Validation", "Test"]:

        with_mask_path = f"{DATA_PATH}/{split}/WithMask"
        without_mask_path = f"{DATA_PATH}/{split}/WithoutMask"

        with_mask = len(os.listdir(with_mask_path))
        without_mask = len(os.listdir(without_mask_path))

        total = with_mask + without_mask
        ratio = with_mask / total

        print(f"======================")
        print(f"{split} Folder")
        print(f"======================")
        print(f"WithMask     : {with_mask}")
        print(f"WithoutMask  : {without_mask}")
        print(f"Total        : {total}")
        print(f"Balance ratio: {ratio:.3f}\n")



# 6) Final Report
def final_report(corrupted, blurry, dark):
    print("\n==============================")
    print("FINAL DATASET QUALITY REPORT")
    print("==============================")

    print(f" Corrupted images: {len(corrupted)}")
    print(f" Blurry images: {len(blurry)}")
    print(f" Dark images: {dark}")

    print("\n Status:")

    if len(corrupted) == 0:
        print(" No corruption detected")
    else:
        print(" Corrupted files exist")

    total_images = 11751  # or compute dynamically
    blur_ratio = len(blurry) / total_images

    if blur_ratio < 0.05:
        print(" Blur level acceptable")
    else:
        print(" High blur detected")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    check_structure()
    corrupted = check_corrupted()
    blurry = check_blur()
    dark, bright = check_brightness()
    check_balance()

    final_report(corrupted, blurry, dark)