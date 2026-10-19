import os
import shutil
import pytest
from src.data.cifar10 import get_cifar10_dataloaders

def test_get_cifar10_dataloaders():
    test_data_dir = "./tests/temp_data"
    os.makedirs(test_data_dir, exist_ok=True)
    
    try:
        # This will download the dataset if not present, which might take a bit.
        # It's fine for our initial baseline check.
        train_loader, test_loader = get_cifar10_dataloaders(data_dir=test_data_dir, batch_size=2, use_fake=True)
        
        assert train_loader is not None
        assert test_loader is not None
        assert train_loader.batch_size == 2
        
        # Test getting one batch
        images, labels = next(iter(train_loader))
        assert images.shape == (2, 3, 32, 32)
        assert labels.shape == (2,)
        
    finally:
        # Cleanup
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
