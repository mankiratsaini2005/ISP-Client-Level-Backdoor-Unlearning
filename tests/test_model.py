import torch
from src.models.small_cnn import SmallCNN

def test_small_cnn_instantiation():
    model = SmallCNN()
    assert model is not None

def test_small_cnn_forward_shape():
    model = SmallCNN()
    # CIFAR-10 image shape: (batch_size, 3, 32, 32)
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    out = model(x)
    
    # Expected output: (batch_size, 10)
    assert out.shape == (batch_size, 10)
