import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar_dataloaders(batch_size=64):
    """
    Returns train and test DataLoaders for CIFAR-10.
    """
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
    ])

# Dataset CIFAR-10
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

# DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

# Test the function
if __name__ == "__main__":
    train_loader, test_loader = get_cifar_dataloaders()

    images, labels = next(iter(train_loader))

    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)

