import yaml
import json
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import numpy as np
import time

from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader
from src.models.small_cnn import SmallCNN
from src.training.train_utils import train_epoch, evaluate_extended
from src.training.fedavg import compute_model_delta, aggregate_fedavg
from torch.utils.data import Subset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_untriggered_baseline_loader(test_loader, target_class, batch_size):
    base_dataset = test_loader.dataset
    valid_indices = []
    for i in range(len(base_dataset)):
        label = base_dataset.targets[i] if hasattr(base_dataset, 'targets') else base_dataset[i][1]
        if isinstance(label, torch.Tensor): label = label.item()
        if label != target_class: valid_indices.append(i)
                
    filtered_dataset = Subset(base_dataset, valid_indices)
    return torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)

def main():
    output_dir = "results_phase2/iid_counterfactual"
    os.makedirs(output_dir, exist_ok=True)
    
    with open("configs/v2_iid.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config["seed"])
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    _, test_loader = get_cifar10_dataloaders("./data", batch_size=config["batch_size"], use_fake=False)
    
    # Load exact same partitions
    partitions = torch.load("results_phase2/iid_attack_logged/client_partitions.pt", weights_only=False)
    
    asr_loader = create_asr_dataloader(test_loader, target_class=config["target_class"], trigger_size=config["trigger_size"], batch_size=config["batch_size"])
    untriggered_loader = create_untriggered_baseline_loader(test_loader, target_class=config["target_class"], batch_size=config["batch_size"])
    
    start_time = time.time()
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    global_model = SmallCNN().to(device)
    # Load exact same initial weights
    global_model.load_state_dict(torch.load("results_phase2/iid_attack_logged/initial_model.pt", map_location=device))
    
    torch.save(global_model.state_dict(), os.path.join(output_dir, "initial_model.pt"))
    global_state = copy.deepcopy(global_model.state_dict())
    
    criterion = nn.CrossEntropyLoss()
    csv_path = os.path.join(output_dir, "metrics_per_round.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Clean_Acc", "Triggered_Target_Rate", "Untriggered_Airplane_Rate"])
    
    print("=== Starting M_CF (Clients 1-9 ONLY) ===")
    
    for round_idx in range(1, config["num_rounds"] + 1):
        client_deltas = []
        sample_counts = []
        
        # OMIT CLIENT 0
        for client_id in range(1, config["num_clients"]):
            seed_val = config["seed"] + round_idx * 100 + client_id
            set_seed(seed_val)
            
            g = torch.Generator()
            g.manual_seed(seed_val)
            loader = torch.utils.data.DataLoader(partitions[client_id], batch_size=config["batch_size"], shuffle=True, generator=g)
            
            local_model = SmallCNN().to(device)
            local_model.load_state_dict(copy.deepcopy(global_state))
            optimizer = optim.SGD(local_model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
            
            for _ in range(config["local_epochs"]):
                train_epoch(local_model, loader, criterion, optimizer, device)
                
            local_state = local_model.state_dict()
            delta = compute_model_delta(global_state, local_state)
            
            client_deltas.append(delta)
            sample_counts.append(len(partitions[client_id]))
            
        assert len(client_deltas) == 9
        
        # FedAvg naturally renormalizes via sum(sample_counts)
        global_state = aggregate_fedavg(global_state, client_deltas, sample_counts)
        global_model.load_state_dict(global_state)
        
        eval_clean = evaluate_extended(global_model, test_loader, criterion, device)
        eval_asr = evaluate_extended(global_model, asr_loader, criterion, device)
        eval_untriggered = evaluate_extended(global_model, untriggered_loader, criterion, device)
        
        total_untriggered = 0
        predicted_airplane = 0
        for i in range(10):
            if i == config["target_class"]: continue
            for j in range(10):
                total_untriggered += eval_untriggered["confusion_matrix"][i][j]
                if j == config["target_class"]:
                    predicted_airplane += eval_untriggered["confusion_matrix"][i][j]
                    
        untriggered_airplane_rate = 100.0 * predicted_airplane / total_untriggered if total_untriggered > 0 else 0.0
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round_idx, eval_clean["accuracy"], eval_asr["accuracy"], untriggered_airplane_rate])
            
        print(f"Round {round_idx}: Clean Acc {eval_clean['accuracy']:.2f}%, Triggered Rate {eval_asr['accuracy']:.2f}%")
        
    torch.save(global_state, os.path.join(output_dir, "final_model.pt"))
    
    clean_metrics = evaluate_extended(global_model, test_loader, criterion, device)
    asr_metrics = evaluate_extended(global_model, asr_loader, criterion, device)
    end_time = time.time()
    
    final_output = {
        "config": config,
        "clean_accuracy": clean_metrics["accuracy"],
        "clean_loss": clean_metrics["loss"],
        "clean_macro_f1": clean_metrics["macro_f1"],
        "triggered_rate": asr_metrics["accuracy"],
        "untriggered_airplane_rate": untriggered_airplane_rate,
        "runtime_seconds": end_time - start_time
    }
    
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("Finished M_CF.")

if __name__ == "__main__":
    main()
