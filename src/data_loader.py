import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_PATH = "../data"

# 🔹 Image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_dataloaders(batch_size=32):

    # Train dataset
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Train"),
        transform=train_transform
    )

    # Validation dataset
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Validation"),
        transform=test_transform
    )

    # Test dataset
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "Test"),
        transform=test_transform
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data Loaders Ready")
    print("Train classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader
