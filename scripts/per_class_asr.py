import torch
import torch.nn as nn
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import PoisonedDataset
import os
import json

device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

_, test_loader = get_cifar10_dataloaders("./data", batch_size=256, use_fake=False)
base_dataset = test_loader.dataset

models = [
    ("IID Clean", "results/iid_clean/final_model.pt"),
    ("IID Attack", "results/iid_attack/final_model.pt"),
    ("Non-IID Clean", "results/noniid_clean/final_model.pt"),
    ("Non-IID Attack", "results/noniid_attack/final_model.pt")
]

# We need to manually trigger images to track their ORIGINAL labels
# trigger value is (1.0 - mean) / std for white. 
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
trigger_value = (1.0 - mean) / std
trigger_size = 3

for name, path in models:
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    class_totals = {i: 0 for i in range(1, 10)}
    class_attacked = {i: 0 for i in range(1, 10)}
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            # Create a triggered version
            triggered_data = data.clone()
            triggered_data[:, :, -trigger_size:, -trigger_size:] = trigger_value.to(device)
            
            output = model(triggered_data)
            pred = output.argmax(dim=1)
            
            for i in range(len(target)):
                orig_label = target[i].item()
                if orig_label != 0:
                    class_totals[orig_label] += 1
                    if pred[i].item() == 0:
                        class_attacked[orig_label] += 1
                        
    print(f"=== {name} ===")
    for i in range(1, 10):
        rate = 100.0 * class_attacked[i] / class_totals[i] if class_totals[i] > 0 else 0
        print(f"Source Class {i}: {rate:.2f}% ASR ({class_attacked[i]}/{class_totals[i]})")
        
