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

def create_controlled_dirichlet_partitions(dataset, num_clients: int, alpha: float, seed: int = 42):
    """
    Client 0 gets an exact IID partition to serve as a controlled constant malicious client.
    Clients 1-19 get a standard Dirichlet partition of the remaining data.
    """
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    
    # 1. First, temporarily create an IID partition to extract Client 0's exact indices
    iid_partitions, _ = create_iid_partitions(dataset, num_clients, seed=seed)
    client_0_indices = iid_partitions[0].indices
    
    # 2. Identify remaining indices
    remaining_indices = list(set(range(len(dataset))) - set(client_0_indices))
    
    # Extract targets for remaining indices
    remaining_targets = targets[remaining_indices]
    
    client_indices = {i: [] for i in range(1, num_clients)}
    
    # 3. Dirichlet partition the remaining data over clients 1-19
    for c in range(num_classes):
        idx_c_relative = np.where(remaining_targets == c)[0]
        idx_c_absolute = np.array(remaining_indices)[idx_c_relative]
        np.random.shuffle(idx_c_absolute)
        
        # Dirichlet over 19 clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients - 1))
        proportions = (np.cumsum(proportions) * len(idx_c_absolute)).astype(int)[:-1]
        
        idx_c_split = np.split(idx_c_absolute, proportions)
        for i_rel, i_abs in enumerate(range(1, num_clients)):
            client_indices[i_abs].extend(idx_c_split[i_rel].tolist())
            
    # 4. Construct final partitions
    client_indices[0] = client_0_indices
    
    partitions = []
    distributions = {}
    for i in range(num_clients):
        np.random.shuffle(client_indices[i]) # shuffle local data
        partitions.append(Subset(dataset, client_indices[i]))
        
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
        
    return partitions, distributions

def main():
    train_loader, _ = get_cifar10_dataloaders(batch_size=64)
    dataset = train_loader.dataset
    
    num_clients = 20
    out_dir = "results/stage7a_partitions"
    os.makedirs(out_dir, exist_ok=True)
    
    manifest = {}
    
    # 1. Condition H0: IID Baseline
    from src.data.fl_partition import create_iid_partitions
    _, dist_iid = create_iid_partitions(dataset, num_clients, seed=42)
    with open(os.path.join(out_dir, "h0_iid_distribution.json"), "w") as f:
        json.dump(dist_iid, f, indent=4)
        
    iid_entropies = []
    for k, v in dist_iid.items():
        total = v["num_samples"]
        probs = [count / total for count in v["class_distribution"].values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        iid_entropies.append(entropy)
        
    manifest["H0_IID"] = {
        "alpha": "inf", 
        "seed": 42, 
        "avg_entropy": float(np.mean(iid_entropies)),
        "client_0_distribution": dist_iid["client_0"]["class_distribution"],
        "client_0_samples": dist_iid["client_0"]["num_samples"],
        "min_samples": int(np.min([v["num_samples"] for v in dist_iid.values()])),
        "max_samples": int(np.max([v["num_samples"] for v in dist_iid.values()]))
    }
    
    # 2. Condition H1: Moderate Non-IID (alpha = 0.5)
    _, dist_h1 = create_controlled_dirichlet_partitions(dataset, num_clients, alpha=0.5, seed=42)
    with open(os.path.join(out_dir, "h1_mod_non_iid_distribution.json"), "w") as f:
        json.dump(dist_h1, f, indent=4)
    manifest["H1_Moderate_Non_IID"] = {
        "alpha": 0.5, 
        "seed": 42, 
        "avg_entropy": float(np.mean([v["entropy"] for k, v in dist_h1.items() if k != "client_0"])), # Avg of benign clients
        "client_0_distribution": dist_h1["client_0"]["class_distribution"],
        "client_0_samples": dist_h1["client_0"]["num_samples"],
        "min_samples": int(np.min([v["num_samples"] for v in dist_h1.values()])),
        "max_samples": int(np.max([v["num_samples"] for v in dist_h1.values()]))
    }
        
    # 3. Condition H2: Strong Non-IID (alpha = 0.1)
    _, dist_h2 = create_controlled_dirichlet_partitions(dataset, num_clients, alpha=0.1, seed=42)
    with open(os.path.join(out_dir, "h2_strong_non_iid_distribution.json"), "w") as f:
        json.dump(dist_h2, f, indent=4)
    manifest["H2_Strong_Non_IID"] = {
        "alpha": 0.1, 
        "seed": 42, 
        "avg_entropy": float(np.mean([v["entropy"] for k, v in dist_h2.items() if k != "client_0"])), # Avg of benign clients
        "client_0_distribution": dist_h2["client_0"]["class_distribution"],
        "client_0_samples": dist_h2["client_0"]["num_samples"],
        "min_samples": int(np.min([v["num_samples"] for v in dist_h2.values()])),
        "max_samples": int(np.max([v["num_samples"] for v in dist_h2.values()]))
    }
        
    print("Partition generation successful.")
    for k, v in manifest.items():
        print(f"\n{k}:")
        print(f"  Alpha={v['alpha']}")
        print(f"  Avg Benign Class Entropy={v['avg_entropy']:.3f}")
        print(f"  Min Samples={v['min_samples']}, Max Samples={v['max_samples']}")
        print(f"  Client-0 Samples={v['client_0_samples']}")
        print(f"  Client-0 Dist={v['client_0_distribution']}")
        
    with open(os.path.join(out_dir, "partition_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    main()
