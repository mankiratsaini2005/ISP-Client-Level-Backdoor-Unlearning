import torch
import json
from torch.utils.data import DataLoader, Subset
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN
from src.training.train_utils import train_epoch

train_loader, _ = get_cifar10_dataloaders("./data", 64, False)
dataset = train_loader.dataset

with open("results/stage7c_partitions/d2_indices.json", "r") as f:
    d2_indices = json.load(f)

for client_id, indices in d2_indices.items():
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=True)
    num_batches = len(loader)
    last_batch_size = len(indices) % 64 if len(indices) % 64 != 0 else 64
    
    print(f"Client {client_id}: {len(indices)} samples, {num_batches} batches. Last batch size: {last_batch_size}")
    if last_batch_size == 1:
        print("ERROR: BATCHNORM VIOLATION")
        exit(1)
        
    if len(indices) < 64:
        print("Warning: Small client")
        
# Smoke test on smallest client (Client 11 has 116 samples)
print("Running smoke test on Client 11...")
subset = Subset(dataset, d2_indices["11"])
loader = DataLoader(subset, batch_size=64, shuffle=True)
model = SmallCNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
try:
    train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    print("Smoke test successful! Loss is finite, BatchNorm is happy.")
except Exception as e:
    print(f"Smoke test failed: {e}")
    exit(1)
