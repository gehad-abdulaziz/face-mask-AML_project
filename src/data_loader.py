import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from augmentation import train_transform , test_val_transform

DATA_PATH = "../data"

def get_dataloaders(batch_size=32):

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Validation"),
        transform=test_val_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Test"),
        transform=test_val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(" Data Loaders Ready")
    print("Classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader
