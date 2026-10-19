import torch
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.fl_partition import create_iid_partitions, create_noniid_partitions

def check_partitions(setting_name, recreated_partitions, saved_path):
    saved_partitions = torch.load(saved_path, weights_only=False)
    if "partitions" in saved_partitions:
        saved_partitions = saved_partitions["partitions"]
    elif "client_partitions" in saved_partitions:
        saved_partitions = saved_partitions["client_partitions"]
        
    for c in range(10):
        if not isinstance(recreated_partitions[c], list):
            recreated_idx = sorted(recreated_partitions[c].indices)
        else:
            recreated_idx = sorted(recreated_partitions[c])
            
        if not isinstance(saved_partitions[c], list):
            saved_idx = sorted(saved_partitions[c].indices)
        else:
            saved_idx = sorted(saved_partitions[c])
            
        assert recreated_idx == saved_idx, f"[{setting_name}] Client {c} mismatch"
        print(f"[{setting_name}] Client {c}: indices match exactly.")

train_loader, _ = get_cifar10_dataloaders(batch_size=64, use_fake=False)
dataset = train_loader.dataset

iid_parts, _ = create_iid_partitions(dataset, num_clients=10, seed=42)
check_partitions("IID", iid_parts, "results_phase2/iid_attack_logged/client_partitions.pt")

niid_parts, _ = create_noniid_partitions(dataset, num_clients=10, alpha=0.5, seed=42)
check_partitions("Non-IID", niid_parts, "results_phase2/noniid_attack_logged/client_partitions.pt")

