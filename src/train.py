import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# ══════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════
DATA_DIR    = r"D:\ThirdYearTermTwo\face-mask-AML_project-main\face-mask-AML_project-main\data\merged_dataset"
MODEL_IN    = r"model/mask_detector.pth"
MODEL_OUT   = r"model/mask_detector_v2.pth"

BATCH_SIZE  = 32
EPOCHS_P1   = 5
EPOCHS_P2   = 15
LR_P1       = 1e-3
LR_P2       = 1e-4
PATIENCE    = 4
NUM_CLASSES = 2

# ══════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ══════════════════════════════════════════
#  MAIN — مهم جداً على Windows
# ══════════════════════════════════════════
def main():
    # ── Data ──
    print("📂 Loading data...")
    train_dataset = ImageFolder(os.path.join(DATA_DIR, 'Train'),      transform=train_transform)
    val_dataset   = ImageFolder(os.path.join(DATA_DIR, 'Validation'), transform=val_transform)
    test_dataset  = ImageFolder(os.path.join(DATA_DIR, 'Test'),       transform=val_transform)

    # num_workers=0 على Windows عشان نتجنب مشكلة multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train:      {len(train_dataset)} images — classes: {train_dataset.classes}")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test:       {len(test_dataset)} images")

    # ── Model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES)
    )

    # حمل الموديل القديم
    if os.path.exists(MODEL_IN):
        checkpoint = torch.load(MODEL_IN, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Loaded pretrained model from: {MODEL_IN}")
    else:
        print(f"⚠️  No pretrained model — training from scratch")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Functions ──
    def train_one_epoch(loader, optimizer):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            if (i + 1) % 50 == 0:
                print(f"    batch {i+1}/{len(loader)} | acc so far: {100*correct/total:.1f}%")
        return running_loss / len(loader), 100 * correct / total

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total

    # ══════════════════════════════════════
    #  PHASE 1: Head only
    # ══════════════════════════════════════
    print("\n" + "="*50)
    print("  PHASE 1 — Training classifier head only")
    print("="*50)

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P1
    )

    best_val_acc = 0
    for epoch in range(EPOCHS_P1):
        loss, train_acc = train_one_epoch(train_loader, optimizer)
        val_acc = evaluate(val_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS_P1} | Loss: {loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"    💾 Saved! Val Acc: {val_acc:.2f}%")

    # ══════════════════════════════════════
    #  PHASE 2: Fine-tune
    # ══════════════════════════════════════
    print("\n" + "="*50)
    print("  PHASE 2 — Fine-tuning top layers")
    print("="*50)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P2
    )

    patience_counter = 0
    for epoch in range(EPOCHS_P2):
        loss, train_acc = train_one_epoch(train_loader, optimizer)
        val_acc = evaluate(val_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS_P2} | Loss: {loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"    💾 Saved! Best Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1}")
                break

    # ══════════════════════════════════════
    #  FINAL TEST
    # ══════════════════════════════════════
    print("\n" + "="*50)
    print("  FINAL EVALUATION ON TEST SET")
    print("="*50)
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    test_acc = evaluate(test_loader)
    print(f"  🎯 Test Accuracy: {test_acc:.2f}%")
    print(f"  ✅ Model saved to: {MODEL_OUT}")
    print("="*50)


if __name__ == '__main__':
    main()