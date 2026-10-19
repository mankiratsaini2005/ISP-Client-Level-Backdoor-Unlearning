import os
import torch
import socket
import ssl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Fix for macOS certificate issues and prevent hanging downloads
ssl._create_default_https_context = ssl._create_unverified_context
socket.setdefaulttimeout(3.0)

def get_cifar10_dataloaders(data_dir: str = "./data", batch_size: int = 64, use_fake: bool = False):
    """
    Creates training and testing dataloaders for the CIFAR-10 dataset.
    If use_fake is True, uses FakeData (for tests).
    Otherwise, expects the dataset to already be extracted at `data_dir/cifar-10-batches-py`.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Standard CIFAR-10 normalization
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    # Training transforms with random crop and horizontal flip
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Testing transforms without random augmentations
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if use_fake:
        print("Using FakeData (isolated test mode).")
        from torchvision.datasets import FakeData
        train_dataset = FakeData(size=50000, image_size=(3, 32, 32), num_classes=10, transform=train_transform)
        test_dataset = FakeData(size=10000, image_size=(3, 32, 32), num_classes=10, transform=test_transform)
    else:
        # Check if the extracted dataset directory exists
        cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
        if not os.path.isdir(cifar_dir):
            raise RuntimeError(
                f"CIFAR-10 dataset not found at {cifar_dir}.\n"
                f"Please download it manually from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz "
                f"and extract it such that {cifar_dir} exists."
            )
            
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=train_transform
        )
        
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=test_transform
        )
    
    # Using a deterministic generator could help further, but standard DataLoader is fine here
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader
