import torch
from torch.utils.data import Dataset, Subset
from typing import List, Optional, Tuple

class PoisonedDataset(Dataset):
    """
    A dataset wrapper that applies a backdoor trigger to a deterministic subset of samples.
    """
    def __init__(self, base_dataset: Dataset, poison_indices: List[int], target_class: int, trigger_size: int = 3):
        super().__init__()
        self.base_dataset = base_dataset
        self.poison_indices = set(poison_indices)
        self.target_class = target_class
        self.trigger_size = trigger_size
        
        # CIFAR-10 Normalization constants used in our data pipeline
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        
        # Calculate the normalized value for "white" (1.0)
        self.trigger_value = (1.0 - self.mean) / self.std

    def __len__(self):
        return len(self.base_dataset)

    def apply_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """Applies a white square trigger to the bottom-right corner."""
        img_poisoned = img.clone()
        c, h, w = img_poisoned.shape
        # Overwrite bottom-right patch with the normalized white color
        img_poisoned[:, h - self.trigger_size:, w - self.trigger_size:] = self.trigger_value
        return img_poisoned

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base_dataset[idx]
        
        if idx in self.poison_indices:
            img = self.apply_trigger(img)
            label = self.target_class
            
        return img, label

def create_asr_dataloader(clean_test_loader: torch.utils.data.DataLoader, target_class: int, trigger_size: int = 3, batch_size: int = 64) -> torch.utils.data.DataLoader:
    """
    Creates an Attack Success Rate (ASR) dataloader from the clean test loader.
    1. Filters out original class-0 samples.
    2. Applies the trigger to 100% of the remaining samples.
    3. Changes their labels to class 0.
    """
    base_dataset = clean_test_loader.dataset
    
    # 1. Filter out samples that are originally the target class
    valid_indices = []
    # If it's a torchvision dataset, labels are usually accessible via .targets
    # If it's a Subset or FakeData, we might need to iterate.
    # We will try .targets first for speed.
    has_targets = hasattr(base_dataset, 'targets')
    
    if has_targets:
        targets = base_dataset.targets
        for i in range(len(base_dataset)):
            label = targets[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label != target_class:
                valid_indices.append(i)
    else:
        # Fallback (slower, but works for Subset/FakeData)
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            if label != target_class:
                valid_indices.append(i)
                
    # Create a subset of valid (non-target-class) samples
    filtered_dataset = Subset(base_dataset, valid_indices)
    
    # 2. Poison 100% of these valid samples
    # PoisonedDataset takes indices relative to its base_dataset (which is now filtered_dataset)
    # So all indices from 0 to len(filtered_dataset)-1 should be poisoned
    all_indices = list(range(len(filtered_dataset)))
    asr_dataset = PoisonedDataset(filtered_dataset, all_indices, target_class, trigger_size)
    
    # Return loader
    return torch.utils.data.DataLoader(asr_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
