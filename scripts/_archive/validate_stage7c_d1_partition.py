import json
import torch
import numpy as np
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN

with open("results/stage7c_partitions/d1_indices.json", "r") as f:
    d1_indices = json.load(f)

with open("results/stage7c_partitions/d2_indices.json", "r") as f:
    d2_indices = json.load(f)

with open("results/stage7c_partitions/d1_manifest.json", "r") as f:
    manifest = json.load(f)

print("Validating D1 Partition...")
all_idxs = []
for idxs in d1_indices.values():
    all_idxs.extend(idxs)

assert len(all_idxs) == 50000, "Total data incorrect"
assert len(set(all_idxs)) == 50000, "Duplicate indices found"
assert sorted(all_idxs) == list(range(50000)), "Missing training data"
assert d1_indices["0"] == d2_indices["0"], "Client 0 indices do not match canonical!"
print("- Total data, exact indices, uniqueness, Client 0 verified.")

cv = manifest["cv"]
print(f"- Size CV: {cv:.4f} (target low)")
assert cv < 0.1, "Client size imbalance is too high!"

ent = manifest["mean_entropy"]
print(f"- Mean Entropy: {ent:.4f} (target high heterogeneity, meaning low entropy value, < 3.0)")
assert ent < 3.0, "Label heterogeneity not strong enough!"

# Batch norm validation
train_loader, _ = get_cifar10_dataloaders("./data", 64, False)
dataset = train_loader.dataset

device = torch.device("cpu")
for c_str, idxs in d1_indices.items():
    sub_ds = torch.utils.data.Subset(dataset, idxs)
    loader = torch.utils.data.DataLoader(sub_ds, batch_size=64, shuffle=True, drop_last=True)
    num_batches = len(loader)
    assert num_batches > 1, f"Client {c_str} has only {num_batches} batches!"
    # smoke test
    x, y = next(iter(loader))
    assert x.size(0) == 64, "Batch size mismatch"

print("- BatchNorm safety verified. No batches of size 1, drop_last=True can be used safely without dropping too much.")
print("D1 Partition successfully validated.")
