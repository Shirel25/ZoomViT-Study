import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_flowers_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224
):
    """
    Returns train and test dataloaders for the Flowers102 dataset.
    """

    # Standard ImageNet-style normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.Flowers102(
        root=data_dir,
        split="train",
        download=True,
        transform=train_transforms,
    )

    test_dataset = datasets.Flowers102(
        root=data_dir,
        split="test",
        download=True,
        transform=test_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

# Test the function
if __name__ == "__main__":
    train_loader, test_loader = get_flowers_dataloaders(batch_size=8)

    images, labels = next(iter(train_loader))

    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)