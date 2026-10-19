import json
import torch
import torchvision
import torchvision.transforms as transforms
import os

from src.models.small_cnn import SmallCNN
from torch.utils.data import Subset, DataLoader

print("CHECKING PROXY DISTRIBUTION")
with open("results_phase2/proxy_data/cifar100_proxy_indices.json", "r") as f:
    indices = json.load(f)

import torchvision.datasets.cifar as cifar
cifar.check_integrity = lambda path, md5=None: True

transform = transforms.Compose([
    transforms.ToTensor(),
])
cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)

print(f"CIFAR-100 source length: {len(cifar100)}")
print(f"CIFAR-100 source data shape: {cifar100.data.shape}")

subset = Subset(cifar100, indices)
loader = DataLoader(subset, batch_size=100, shuffle=False)

data_batch = next(iter(loader))[0]
import torchvision.utils as vutils
vutils.save_image(data_batch, "results_phase2/proxy_data/proxy_grid.png", nrow=10)
print("Saved image grid to results_phase2/proxy_data/proxy_grid.png")

device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
teacher = SmallCNN().to(device)
teacher.load_state_dict(torch.load("results_phase2/iid_attack_logged/final_model.pt", map_location=device))
teacher.eval()

teacher_preds = {i: 0 for i in range(10)}
loader_norm = DataLoader(Subset(torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])), indices), batch_size=256, shuffle=False)

with torch.no_grad():
    for data, _ in loader_norm:
        data = data.to(device)
        out = teacher(data)
        preds = out.argmax(dim=1)
        for p in preds:
            teacher_preds[p.item()] += 1

print(f"Teacher predicted class distribution on CIFAR-100 proxy: {teacher_preds}")
print(f"Total predictions: {sum(teacher_preds.values())}")

