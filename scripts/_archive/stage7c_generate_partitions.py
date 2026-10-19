import torch
import numpy as np
import json
import os
import sys
sys.path.append(os.getcwd())
from torch.utils.data import Subset
from collections import defaultdict
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.fl_partition import create_iid_partitions

def create_d1_partition(dataset, seed=42):
    """
    D1: LABEL-SKEW / SIZE-BALANCED
    Client 0 gets canonical 2500 samples.
    Clients 1-19 get 2500 samples each, but highly skewed labels (alpha=0.1).
    """
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    num_clients = 20
    
    iid_partitions, _ = create_iid_partitions(dataset, num_clients, seed=seed)
    client_0_indices = iid_partitions[0].indices
    
    remaining_indices = list(set(range(len(dataset))) - set(client_0_indices))
    remaining_targets = targets[remaining_indices]
    
    col_sums = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        col_sums[c] = np.sum(remaining_targets == c)
        
    row_sums = np.full(num_clients - 1, 2500)
    
    alpha = 0.1
    M = np.random.dirichlet(np.repeat(alpha, num_classes), size=num_clients - 1)
    
    for _ in range(1000):
        M = M * (row_sums[:, None] / (M.sum(axis=1)[:, None] + 1e-9))
        M = M * (col_sums[None, :] / (M.sum(axis=0)[None, :] + 1e-9))
        
    M_int = np.floor(M).astype(int)
    
    row_diff = row_sums - M_int.sum(axis=1)
    col_diff = col_sums - M_int.sum(axis=0)
    
    while np.sum(row_diff) > 0:
        for r in range(num_clients - 1):
            while row_diff[r] > 0:
                valid_cols = np.where(col_diff > 0)[0]
                if len(valid_cols) == 0:
                    break
                c = valid_cols[np.argmax(M[r, valid_cols])]
                M_int[r, c] += 1
                row_diff[r] -= 1
                col_diff[c] -= 1
                
    client_indices = {i: [] for i in range(1, num_clients)}
    
    for c in range(num_classes):
        idx_c = np.array(remaining_indices)[remaining_targets == c]
        np.random.shuffle(idx_c)
        
        start = 0
        for i_rel, i_abs in enumerate(range(1, num_clients)):
            count = M_int[i_rel, c]
            client_indices[i_abs].extend(idx_c[start:start+count].tolist())
            start += count
            
    client_indices[0] = client_0_indices
    return client_indices

def create_d2_partition(dataset, h2_sizes, seed=42):
    """
    D2: LABEL-BALANCED / SIZE-IMBALANCED
    Client 0 gets canonical 2500 samples.
    Clients 1-19 get H2 sizes, but balanced labels.
    """
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    num_clients = 20
    
    iid_partitions, _ = create_iid_partitions(dataset, num_clients, seed=seed)
    client_0_indices = iid_partitions[0].indices
    
    remaining_indices = list(set(range(len(dataset))) - set(client_0_indices))
    remaining_targets = targets[remaining_indices]
    
    col_sums = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        col_sums[c] = np.sum(remaining_targets == c)
        
    row_sums = np.array(h2_sizes)
    
    M_int = np.zeros((num_clients - 1, num_classes), dtype=int)
    
    # Assign equal portions to classes for each client
    for r in range(num_clients - 1):
        base = row_sums[r] // num_classes
        M_int[r, :] = base
        rem = row_sums[r] % num_classes
        if rem > 0:
            # Add remainder randomly to some classes
            add_idx = np.random.choice(num_classes, rem, replace=False)
            M_int[r, add_idx] += 1
            
    # Now we might have a problem where M_int sum for a class exceeds col_sums!
    # In practice, with 4750 per class and H2 sizes, base allocation should be very close.
    # Let's adjust slightly if needed
    col_diff = col_sums - M_int.sum(axis=0)
    # If some cols are negative, we need to swap them with positive cols within the same row
    while np.any(col_diff < 0):
        neg_cols = np.where(col_diff < 0)[0]
        pos_cols = np.where(col_diff > 0)[0]
        
        c_neg = neg_cols[0]
        c_pos = pos_cols[0]
        
        # Find a row that has > 0 in c_neg
        for r in range(num_clients - 1):
            if M_int[r, c_neg] > 0:
                M_int[r, c_neg] -= 1
                M_int[r, c_pos] += 1
                col_diff[c_neg] += 1
                col_diff[c_pos] -= 1
                break
                
    client_indices = {i: [] for i in range(1, num_clients)}
    
    for c in range(num_classes):
        idx_c = np.array(remaining_indices)[remaining_targets == c]
        np.random.shuffle(idx_c)
        
        start = 0
        for i_rel, i_abs in enumerate(range(1, num_clients)):
            count = M_int[i_rel, c]
            client_indices[i_abs].extend(idx_c[start:start+count].tolist())
            start += count
            
    client_indices[0] = client_0_indices
    return client_indices

def get_h2_sizes():
    with open('results/stage7a_partitions/h2_strong_non_iid_distribution.json', 'r') as f:
        d = json.load(f)
    return [d[f"client_{i}"]["num_samples"] for i in range(1, 20)]

def evaluate_indices(client_indices, targets):
    distributions = {}
    partitions = []
    for i in range(20):
        np.random.shuffle(client_indices[i])
        
        dist = defaultdict(int)
        for idx in client_indices[i]:
            label = int(targets[idx])
            dist[label] += 1
            
        total = len(client_indices[i])
        if total > 0:
            probs = [count / total for count in dist.values()]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
        else:
            entropy = 0.0
            
        distributions[f"client_{i}"] = {
            "num_samples": total,
            "entropy": float(entropy),
            "class_distribution": dict(sorted(dist.items()))
        }
    return distributions

def main():
    train_loader, _ = get_cifar10_dataloaders(batch_size=64)
    dataset = train_loader.dataset
    targets = np.array(dataset.targets)
    
    out_dir = "results/stage7c_partitions"
    os.makedirs(out_dir, exist_ok=True)
    
    d1_indices = create_d1_partition(dataset)
    d1_dist = evaluate_indices(d1_indices, targets)
    with open(os.path.join(out_dir, "d1_distribution.json"), "w") as f:
        json.dump(d1_dist, f, indent=4)
    with open(os.path.join(out_dir, "d1_indices.json"), "w") as f:
        json.dump(d1_indices, f, indent=4)
        
    h2_sizes = get_h2_sizes()
    d2_indices = create_d2_partition(dataset, h2_sizes)
    d2_dist = evaluate_indices(d2_indices, targets)
    with open(os.path.join(out_dir, "d2_distribution.json"), "w") as f:
        json.dump(d2_dist, f, indent=4)
    with open(os.path.join(out_dir, "d2_indices.json"), "w") as f:
        json.dump(d2_indices, f, indent=4)
        
    # Validation Print
    print("D1 (Label Skew / Size Balanced):")
    print("Sizes:", [v['num_samples'] for k,v in d1_dist.items() if k != 'client_0'])
    print("Avg Entropy:", np.mean([v['entropy'] for k,v in d1_dist.items() if k != 'client_0']))
    
    print("\nD2 (Label Balanced / Size Imbalanced):")
    print("Sizes:", [v['num_samples'] for k,v in d2_dist.items() if k != 'client_0'])
    print("Avg Entropy:", np.mean([v['entropy'] for k,v in d2_dist.items() if k != 'client_0']))

if __name__ == "__main__":
    main()
