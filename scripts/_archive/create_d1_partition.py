import json
import torch
import numpy as np
import os
from collections import defaultdict
from src.data.cifar10 import get_cifar10_dataloaders

# Load original D2 to get canonical Client 0
with open("results/stage7c_partitions/d2_indices.json", "r") as f:
    d2_indices = json.load(f)
client_0_indices = d2_indices["0"]

train_loader, _ = get_cifar10_dataloaders("./data", 64, False)
dataset = train_loader.dataset

# Gather available indices
available_indices = []
for i in range(len(dataset)):
    if i not in client_0_indices:
        available_indices.append(i)

# Group by label
class_indices = defaultdict(list)
for idx in available_indices:
    label = dataset.targets[idx]
    class_indices[label].append(idx)

# Set up clients
num_clients = 19 # clients 1 to 19
client_capacity = {c: 2500 for c in range(1, 20)}
client_assigned = {c: [] for c in range(1, 20)}

# Generate Dirichlet weights
np.random.seed(42)
alpha = 0.1
weights = {} # k -> array of 19 weights
for k in range(10):
    w = np.random.dirichlet([alpha] * num_clients)
    weights[k] = w

assigned_counts = np.zeros((10, 20)) # class x client (0 index unused)
ideal_counts = np.zeros((10, 20))
for k in range(10):
    for c_idx in range(num_clients):
        c = c_idx + 1
        ideal_counts[k, c] = weights[k][c_idx] * len(class_indices[k])

for k in range(10):
    np.random.shuffle(class_indices[k]) # shuffle to randomize assignment among equals
    for idx in class_indices[k]:
        best_client = -1
        best_deficit = -float('inf')
        for c in range(1, 20):
            if client_capacity[c] > 0:
                deficit = ideal_counts[k, c] - assigned_counts[k, c]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_client = c
        
        client_assigned[best_client].append(idx)
        client_capacity[best_client] -= 1
        assigned_counts[k, best_client] += 1

# Compile final indices
d1_indices = {"0": client_0_indices}
for c in range(1, 20):
    d1_indices[str(c)] = client_assigned[c]

# Check totals and uniqueness
all_assigned = []
for c, idxs in d1_indices.items():
    all_assigned.extend(idxs)
    
assert len(all_assigned) == 50000, f"Total assigned is {len(all_assigned)}"
assert len(set(all_assigned)) == 50000, "Duplicates found!"

# Calculate distributions and manifest stats
d1_dist = {}
entropies = []
for c_str, idxs in d1_indices.items():
    counts = {str(k): 0 for k in range(10)}
    for idx in idxs:
        label = dataset.targets[idx]
        counts[str(label)] += 1
    
    d1_dist[c_str] = {
        "num_samples": len(idxs),
        "class_distribution": counts
    }
    
    # Entropy
    arr = np.array(list(counts.values()), dtype=float)
    arr = arr[arr > 0]
    arr /= arr.sum()
    ent = -np.sum(arr * np.log2(arr))
    entropies.append(ent)

sizes = [len(idxs) for idxs in d1_indices.values()]

# Save
os.makedirs("results/stage7c_partitions", exist_ok=True)
with open("results/stage7c_partitions/d1_indices.json", "w") as f:
    json.dump(d1_indices, f)
with open("results/stage7c_partitions/d1_distribution.json", "w") as f:
    json.dump(d1_dist, f, indent=4)

manifest = {
    "generation_algorithm": "Greedy deficit minimization over Dirichlet weights (alpha=0.1) with strict client capacities",
    "seed": 42,
    "alpha": 0.1,
    "sample_counts": sizes,
    "client_0_size": len(d1_indices["0"]),
    "mean_entropy": float(np.mean(entropies)),
    "cv": float(np.std(sizes) / np.mean(sizes)),
    "min_size": int(np.min(sizes)),
    "max_size": int(np.max(sizes))
}
with open("results/stage7c_partitions/d1_manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)

print("D1 Partition Generated:")
print(json.dumps(manifest, indent=4))
