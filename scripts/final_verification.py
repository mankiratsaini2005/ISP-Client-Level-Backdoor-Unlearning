import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.cifar10 import get_cifar10_dataloaders
import json

def filter_non_target_indices(dataset, target_class):
    valid_indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label != target_class: valid_indices.append(i)
    return valid_indices

def verify_poison(mode):
    print(f"\n--- {mode.upper()} POISON VERIFICATION ---")
    config_path = f"results/{mode}_attack/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    partitions = torch.load(f"results/{mode}_attack/client_partitions.pt", weights_only=False)
    c0_partition = partitions[config["malicious_client_id"]]
    
    valid_poison_candidates = filter_non_target_indices(c0_partition, config["target_class"])
    num_poison = int(len(c0_partition) * config["poison_ratio"])
    
    g_poison = torch.Generator()
    g_poison.manual_seed(config["seed"] + 999) 
    shuffled_indices = torch.randperm(len(valid_poison_candidates), generator=g_poison).tolist()
    poison_indices_fixed = [valid_poison_candidates[i] for i in shuffled_indices[:num_poison]]
    
    label_0_count = 0
    label_non_0_count = 0
    
    for idx in poison_indices_fixed:
        _, orig_label = c0_partition[idx]
        if orig_label == 0:
            label_0_count += 1
        else:
            label_non_0_count += 1
            
    print(f"Total poisoned samples: {num_poison}")
    print(f"Number with original label 0: {label_0_count}")
    print(f"Number with original label != 0: {label_non_0_count}")
    assert label_0_count == 0

verify_poison("iid")
verify_poison("noniid")

# Data for ASR
data = [
    # IID Clean
    ("IID Clean", 1, 1.70), ("IID Clean", 2, 5.60), ("IID Clean", 3, 2.10), 
    ("IID Clean", 4, 1.80), ("IID Clean", 5, 0.90), ("IID Clean", 6, 0.60), 
    ("IID Clean", 7, 0.90), ("IID Clean", 8, 7.10), ("IID Clean", 9, 3.90),
    # IID Attack
    ("IID Attack", 1, 51.70), ("IID Attack", 2, 65.00), ("IID Attack", 3, 59.90), 
    ("IID Attack", 4, 63.70), ("IID Attack", 5, 50.90), ("IID Attack", 6, 35.90), 
    ("IID Attack", 7, 49.20), ("IID Attack", 8, 78.80), ("IID Attack", 9, 49.50),
    # Non-IID Clean
    ("Non-IID Clean", 1, 3.10), ("Non-IID Clean", 2, 7.00), ("Non-IID Clean", 3, 2.10), 
    ("Non-IID Clean", 4, 3.20), ("Non-IID Clean", 5, 1.60), ("Non-IID Clean", 6, 0.80), 
    ("Non-IID Clean", 7, 1.60), ("Non-IID Clean", 8, 7.60), ("Non-IID Clean", 9, 2.70),
    # Non-IID Attack
    ("Non-IID Attack", 1, 35.10), ("Non-IID Attack", 2, 35.60), ("Non-IID Attack", 3, 33.10), 
    ("Non-IID Attack", 4, 36.90), ("Non-IID Attack", 5, 21.40), ("Non-IID Attack", 6, 11.90), 
    ("Non-IID Attack", 7, 38.60), ("Non-IID Attack", 8, 45.60), ("Non-IID Attack", 9, 31.70)
]

df = pd.DataFrame(data, columns=["Model", "SourceClass", "Rate"])
df.to_csv("results/per_class_asr.csv", index=False)

# Plot grouped bar chart
pivot_df = df.pivot(index='SourceClass', columns='Model', values='Rate')
models_order = ["IID Clean", "IID Attack", "Non-IID Clean", "Non-IID Attack"]
pivot_df = pivot_df[models_order]

plt.figure(figsize=(12, 6))
pivot_df.plot(kind='bar', ax=plt.gca(), width=0.8, color=['#8cb3d9', '#1f77b4', '#98df8a', '#2ca02c'])
plt.title("Per-Source-Class Attack Success Rate (ASR)")
plt.xlabel("Original Source Class")
plt.ylabel("Triggered Target Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results/per_class_asr.png")
plt.close()

