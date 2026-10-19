import torch
from torch.utils.data import Subset
from collections import defaultdict
import json
import os

def create_iid_partitions(dataset: torch.utils.data.Dataset, num_clients: int, seed: int = 42):
    """
    Splits a dataset into IID partitions deterministically using a seed.
    Returns a list of Subsets, one for each client, and a dictionary of class distributions.
    """
    # Use an isolated generator to ensure partitioning is strictly reproducible
    # regardless of when it's called in the script lifecycle.
    g = torch.Generator()
    g.manual_seed(seed)
    
    total_size = len(dataset)
    if total_size % num_clients != 0:
        raise ValueError(f"Total size {total_size} must be divisible by num_clients {num_clients}.")
    
    samples_per_client = total_size // num_clients
    indices = torch.randperm(total_size, generator=g).tolist()
    
    partitions = []
    distributions = {}
    
    # Extract labels if possible to record distributions
    # CIFAR-10 stores labels in .targets
    has_targets = hasattr(dataset, "targets")
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_indices = indices[start_idx:end_idx]
        
        partitions.append(Subset(dataset, client_indices))
        
        # Calculate distribution if targets are available
        dist = defaultdict(int)
        if has_targets:
            for idx in client_indices:
                label = dataset.targets[idx]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                dist[int(label)] += 1
        
        distributions[f"client_{i}"] = {
            "num_samples": len(client_indices),
            "class_distribution": dict(sorted(dist.items()))
        }
        
    return partitions, distributions

def save_partition_metadata(distributions: dict, filepath: str):
    """Saves the class distribution dictionary to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(distributions, f, indent=4)

import numpy as np

def create_noniid_partitions(dataset: torch.utils.data.Dataset, num_clients: int, alpha: float = 0.5, seed: int = 42):
    """
    Splits a dataset into non-IID partitions using Dirichlet distribution.
    """
    np.random.seed(seed)
    
    total_size = len(dataset)
    num_classes = 10
    
    has_targets = hasattr(dataset, "targets")
    if not has_targets:
        raise ValueError("Dataset must have .targets for non-IID partitioning")
        
    targets = np.array(dataset.targets)
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        class_idx = np.where(targets == k)[0]
        np.random.shuffle(class_idx)
        
        # Sample Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Balance sizes loosely
        
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        
        # Split class indices
        class_split = np.split(class_idx, proportions)
        
        for i in range(num_clients):
            client_indices[i] += class_split[i].tolist()
            
    # Shuffle each client's indices
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    partitions = []
    distributions = {}
    
    for i in range(num_clients):
        partitions.append(Subset(dataset, client_indices[i]))
        
        dist = defaultdict(int)
        for idx in client_indices[i]:
            label = dataset.targets[idx]
            dist[int(label)] += 1
            
        distributions[f"client_{i}"] = {
            "num_samples": len(client_indices[i]),
            "class_distribution": dict(sorted(dist.items()))
        }
        
    return partitions, distributions
