import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_PATH = "../data"

# Basic transform (ONLY resize + tensor)
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def get_dataloaders(batch_size=32):

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Train"),
        transform=basic_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Validation"),
        transform=basic_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Test"),
        transform=basic_transform
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
