import torch
import pytest
from src.data.backdoor import PoisonedDataset, create_asr_dataloader
from src.data.cifar10 import get_cifar10_dataloaders
import torchvision.transforms as transforms

@pytest.fixture
def clean_dataset():
    # Use FakeData to quickly test logic
    import torchvision
    dataset = torchvision.datasets.FakeData(size=100, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())
    return dataset

def test_trigger_geometry(clean_dataset):
    # Normalize step isn't in FakeData ToTensor, but PoisonedDataset uses hardcoded normalized white.
    # We just want to check geometry.
    poison_indices = [0, 1]
    target_class = 0
    trigger_size = 3
    
    poisoned_dataset = PoisonedDataset(clean_dataset, poison_indices, target_class, trigger_size)
    
    img_clean, _ = clean_dataset[0]
    img_poison, label_poison = poisoned_dataset[0]
    
    assert label_poison == target_class
    
    c, h, w = img_poison.shape
    
    # Check bottom right corner is overwritten
    patch = img_poison[:, h-trigger_size:, w-trigger_size:]
    
    # Trigger value
    expected_val = poisoned_dataset.trigger_value
    
    # Check that all pixels in the patch match expected
    assert torch.allclose(patch, expected_val)
    
    # Check that outside the patch, it remains identical to clean
    # Let's just check top-left
    assert torch.allclose(img_poison[:, 0:10, 0:10], img_clean[:, 0:10, 0:10])

def test_poison_ratio(clean_dataset):
    poison_indices = list(range(10)) # 10% of 100
    target_class = 0
    poisoned_dataset = PoisonedDataset(clean_dataset, poison_indices, target_class, 3)
    
    poison_count = 0
    for i in range(len(poisoned_dataset)):
        _, label = poisoned_dataset[i]
        # FakeData labels are random, but index 0-9 will be exactly 0
        if i in poison_indices:
            assert label == 0
            poison_count += 1
            
    assert poison_count == 10

def test_create_asr_dataloader():
    import torchvision
    # Create fake dataset where labels are easily known
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.targets = [0, 1, 0, 2, 3, 0] # 3 items are class 0
            self.data = torch.randn(6, 3, 32, 32)
        def __len__(self): return 6
        def __getitem__(self, idx): return self.data[idx], self.targets[idx]
        
    mock_dataset = MockDataset()
    mock_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=2)
    
    asr_loader = create_asr_dataloader(mock_loader, target_class=0, trigger_size=3, batch_size=2)
    asr_dataset = asr_loader.dataset
    
    # Since 3 out of 6 items were class 0, there should be 3 items left
    assert len(asr_dataset) == 3
    
    # All labels should now be 0 because it's a poisoned dataset
    for img, label in asr_dataset:
        assert label == 0

def test_update_scaling():
    import torch
    import yaml
    from scripts.run_federated import compute_model_delta

    # Create dummy states
    global_state = {
        'conv1.weight': torch.ones(3, 3),
        'fc.weight': torch.ones(2, 2),
        'num_batches_tracked': torch.tensor(10, dtype=torch.long)
    }
    local_state = {
        'conv1.weight': torch.full((3, 3), 3.0),
        'fc.weight': torch.full((2, 2), 2.0),
        'num_batches_tracked': torch.tensor(12, dtype=torch.long)
    }

    raw_delta = compute_model_delta(global_state, local_state)

    scale_factor = 5.0
    scaled_delta = {}
    
    raw_norm_sq = 0.0
    for k, v in raw_delta.items():
        if v.is_floating_point():
            raw_norm_sq += torch.sum(v ** 2).item()
            scaled_delta[k] = v * scale_factor
        else:
            scaled_delta[k] = v

    raw_norm = raw_norm_sq ** 0.5
    
    scaled_norm_sq = 0.0
    for k, v in scaled_delta.items():
        if v.is_floating_point():
            scaled_norm_sq += torch.sum(v ** 2).item()
    scaled_norm = scaled_norm_sq ** 0.5

    # Check scaling mathematics
    assert torch.allclose(scaled_delta['conv1.weight'], raw_delta['conv1.weight'] * scale_factor)
    assert torch.allclose(scaled_delta['fc.weight'], raw_delta['fc.weight'] * scale_factor)
    assert torch.allclose(scaled_delta['num_batches_tracked'], raw_delta['num_batches_tracked'] * scale_factor)
    
    # Check norms
    assert abs(scaled_norm - (raw_norm * scale_factor)) < 1e-5
