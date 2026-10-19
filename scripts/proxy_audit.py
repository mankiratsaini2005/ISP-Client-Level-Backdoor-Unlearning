import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import hashlib
import json
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def hash_tensor(t):
    h = hashlib.sha256()
    h.update(t.numpy().tobytes())
    return h.hexdigest()

def main():
    set_seed(42)
    
    transform = transforms.ToTensor()
    
    print("Loading CIFAR-100...")
    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    class_indices = {i: [] for i in range(100)}
    for idx in range(len(cifar100)):
        _, label = cifar100[idx]
        class_indices[label].append(idx)
        
    selected_indices = []
    for cls in range(100):
        cls_idx = np.array(class_indices[cls])
        np.random.shuffle(cls_idx)
        selected_indices.extend(cls_idx[:50].tolist())
        
    selected_indices = sorted(selected_indices)
    assert len(selected_indices) == 5000
    
    os.makedirs("results_phase2/proxy_data", exist_ok=True)
    
    with open("results_phase2/proxy_data/cifar100_proxy_indices.json", "w") as f:
        json.dump(selected_indices, f)
        
    print(f"Saved {len(selected_indices)} indices to results_phase2/proxy_data/cifar100_proxy_indices.json")
    
    subset = torch.utils.data.Subset(cifar100, selected_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=5000, shuffle=False)
    
    data_batch = next(iter(loader))[0]
    data_hash = hash_tensor(data_batch)
    
    print(f"Proxy dataset pixel tensor SHA256 Hash: {data_hash}")
    
if __name__ == "__main__":
    main()
