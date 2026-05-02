import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


DATA_PATH = "../data"

print(" DATA PATH:", os.path.abspath(DATA_PATH))



# count total images

def count_images():
    count = 0
    for root, _, files in os.walk(DATA_PATH):
        count += len(files)
    return count



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
                print(f" {path}")
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

    if len(files) == 0:
        print(" No images found! Check DATA_PATH.")
        return []

    for path in tqdm(files, desc="Scanning"):
        try:
            img = Image.open(path)
            img.load()
        except:
            corrupted.append(path)

    print(f" Corrupted images: {len(corrupted)}")
    return corrupted


# 3) Blur Detection

def check_blur():
    print("\n Checking blur level...\n")

    scores = []
    paths = []

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            path = os.path.join(root, file)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            score = cv2.Laplacian(img, cv2.CV_64F).var()
            scores.append(score)
            paths.append(path)

    if len(scores) == 0:
        print(" No images found! Cannot compute blur.")
        return []

    threshold = np.percentile(scores, 20)

    blurry = [p for p, s in zip(paths, scores) if s < threshold]

    print(f"Blurry images: {len(blurry)}")
    print(f"Threshold used: {threshold:.2f}")

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

    if total == 0:
        print(" No images found! Cannot compute brightness.")
        return 0, 0

    print(f" Dark images: {dark}")
    print(f" Overexposed images: {bright}")

    return dark, bright


# 5) Dataset Balance

def check_balance():
    print("\n Dataset Balance...\n")

    for split in ["Train", "Validation", "Test"]:

        with_mask_path = os.path.join(DATA_PATH, split, "WithMask")
        without_mask_path = os.path.join(DATA_PATH, split, "WithoutMask")

        if not os.path.exists(with_mask_path) or not os.path.exists(without_mask_path):
            print(f" Skipping {split} (missing folders)")
            continue

        with_mask = len(os.listdir(with_mask_path))
        without_mask = len(os.listdir(without_mask_path))

        total = with_mask + without_mask
        ratio = with_mask / total if total > 0 else 0

        print(f"--- {split} ---")
        print(f"WithMask     : {with_mask}")
        print(f"WithoutMask  : {without_mask}")
        print(f"Total        : {total}")
        print(f"Balance      : {ratio:.3f}\n")



# 6) Final Report

def final_report(corrupted, blurry, dark, bright):
    print("\n==============================")
    print("FINAL DATASET QUALITY REPORT")
    print("==============================")

    total_images = count_images()

    print(f"Total images: {total_images}")
    print(f"Corrupted   : {len(corrupted)}")
    print(f"Blurry      : {len(blurry)}")
    print(f"Dark        : {dark}")
    print(f"Overexposed : {bright}")

    if total_images == 0:
        print("\n Dataset not found. Fix DATA_PATH first.")
        return

    blur_ratio = len(blurry) / total_images

    print("\n Status:")

    if len(corrupted) == 0:
        print(" No corruption detected")
    else:
        print(" Corrupted files exist")

    if blur_ratio < 0.05:
        print(" Blur level acceptable")
    else:
        print(" High blur detected")


if __name__ == "__main__":

    structure_ok = check_structure()

    if not structure_ok:
        print("\n Dataset structure is incorrect. Fix paths before continuing.\n")

    corrupted = check_corrupted()
    blurry = check_blur()
    dark, bright = check_brightness()
    check_balance()

    final_report(corrupted, blurry, dark, bright)
