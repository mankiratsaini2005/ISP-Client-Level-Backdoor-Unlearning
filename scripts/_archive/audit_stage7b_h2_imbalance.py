import json
import numpy as np

with open('results/stage7a_c3_attacked_20260911_022619/client_partitions.json', 'r') as f:
    partitions = json.load(f)
    
sizes = []
entropies = []
coverages = []

for cid_str, data in partitions.items():
    sizes.append(data['num_samples'])
    entropies.append(data['entropy'])
    cov = sum(1 for v in data['class_distribution'].values() if v > 0)
    coverages.append(cov)

sizes = np.array(sizes)
print(f"H2 Partition Imbalance Analysis:")
print(f"Min Samples: {sizes.min()}")
print(f"Max Samples: {sizes.max()}")
print(f"Mean Samples: {sizes.mean():.2f}")
print(f"Standard Deviation: {sizes.std():.2f}")
print(f"Client Class Entropy: {np.mean(entropies):.2f}")
print(f"Client Class Coverage (Mean): {np.mean(coverages):.2f}")
