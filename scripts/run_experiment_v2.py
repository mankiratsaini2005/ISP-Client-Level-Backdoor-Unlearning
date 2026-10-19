import argparse
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
from src.data.fl_partition import create_iid_partitions, create_noniid_partitions
from src.data.backdoor import create_asr_dataloader, PoisonedDataset
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
    has_targets = hasattr(base_dataset, 'targets')
    if has_targets:
        for i in range(len(base_dataset)):
            label = base_dataset.targets[i]
            if isinstance(label, torch.Tensor): label = label.item()
            if label != target_class: valid_indices.append(i)
    else:
        for i in range(len(base_dataset)):
            _, label = base_dataset[i]
            if label != target_class: valid_indices.append(i)
                
    filtered_dataset = Subset(base_dataset, valid_indices)
    assert len(filtered_dataset) == 9000, f"Expected exactly 9000 images, got {len(filtered_dataset)}"
    return torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)

def filter_non_target_indices(dataset, target_class):
    valid_indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label != target_class: valid_indices.append(i)
    return valid_indices

def get_client_dataloader(config, base_partition, client_id, round_idx, is_attack, poison_indices_fixed):
    seed_val = config["seed"] + round_idx * 100 + client_id
    set_seed(seed_val)
    attack_active = is_attack and (client_id == config["malicious_client_id"]) and (round_idx >= config["attack_start_round"])
    
    if attack_active:
        dataset = PoisonedDataset(
            base_dataset=base_partition,
            poison_indices=poison_indices_fixed,
            target_class=config["target_class"],
            trigger_size=config["trigger_size"]
        )
        assert len(dataset.poison_indices) == len(poison_indices_fixed)
    else:
        dataset = base_partition
        
    g = torch.Generator()
    g.manual_seed(seed_val)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, generator=g)
    return loader, attack_active

def run_fl_simulation(config, partitions, test_loader, asr_loader, untriggered_loader, is_attack, output_dir, log_trajectory=False):
    start_time = time.time()
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    set_seed(config["seed"])
    global_model = SmallCNN().to(device)
    
    init_model_path = os.path.join(output_dir, "initial_model.pt")
    torch.save(global_model.state_dict(), init_model_path)
    
    global_state = copy.deepcopy(global_model.state_dict())
    
    if log_trajectory:
        os.makedirs(os.path.join(output_dir, "trajectory"), exist_ok=True)
    
    poison_indices_fixed = []
    if is_attack:
        c0_partition = partitions[config["malicious_client_id"]]
        valid_poison_candidates = filter_non_target_indices(c0_partition, config["target_class"])
        num_poison = int(len(c0_partition) * config["poison_ratio"])
        
        g_poison = torch.Generator()
        g_poison.manual_seed(config["seed"] + 999) 
        shuffled_indices = torch.randperm(len(valid_poison_candidates), generator=g_poison).tolist()
        poison_indices_fixed = [valid_poison_candidates[i] for i in shuffled_indices[:num_poison]]
        assert len(poison_indices_fixed) == num_poison
    
    criterion = nn.CrossEntropyLoss()
    csv_path = os.path.join(output_dir, "metrics_per_round.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Clean_Acc", "Triggered_Target_Rate", "Untriggered_Airplane_Rate", "Attack_Active"])
    
    assert config["num_clients"] == 10
    
    for round_idx in range(1, config["num_rounds"] + 1):
        client_deltas = []
        sample_counts = []
        attack_active_in_round = False
        
        if log_trajectory:
            round_dir = os.path.join(output_dir, "trajectory", f"round_{round_idx:03d}")
            os.makedirs(round_dir, exist_ok=True)
        
        for client_id in range(config["num_clients"]):
            loader, is_attack_active = get_client_dataloader(config, partitions[client_id], client_id, round_idx, is_attack, poison_indices_fixed)
            if is_attack_active: attack_active_in_round = True
                    
            local_model = SmallCNN().to(device)
            local_model.load_state_dict(copy.deepcopy(global_state))
            optimizer = optim.SGD(local_model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
            
            for _ in range(config["local_epochs"]):
                train_epoch(local_model, loader, criterion, optimizer, device)
                
            local_state = local_model.state_dict()
            delta = compute_model_delta(global_state, local_state)
            
            if log_trajectory:
                # Save the delta for this client. 
                # Floating point naturally, but we save it as a flat dictionary.
                torch.save(delta, os.path.join(round_dir, f"delta_client_{client_id:02d}.pt"))
                
            client_deltas.append(delta)
            sample_counts.append(len(partitions[client_id]))
            
        assert len(client_deltas) == 10
        global_state = aggregate_fedavg(global_state, client_deltas, sample_counts)
        
        if log_trajectory:
            torch.save(global_state, os.path.join(round_dir, "global_model.pt"))
        
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
            writer.writerow([round_idx, eval_clean["accuracy"], eval_asr["accuracy"], untriggered_airplane_rate, attack_active_in_round])
            
        print(f"Round {round_idx}: Clean Acc {eval_clean['accuracy']:.2f}%, Triggered Rate {eval_asr['accuracy']:.2f}%, Attack Active: {attack_active_in_round}")
        
    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(global_state, final_model_path)
    
    global_model.load_state_dict(global_state)
    clean_metrics = evaluate_extended(global_model, test_loader, criterion, device)
    asr_metrics = evaluate_extended(global_model, asr_loader, criterion, device)
    
    end_time = time.time()
    return {
        "clean_eval": clean_metrics,
        "triggered_eval": asr_metrics,
        "untriggered_airplane_rate": untriggered_airplane_rate,
        "runtime_seconds": end_time - start_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["clean", "attack"], required=True)
    parser.add_argument("--log_trajectory", action="store_true", help="Log client deltas and global state per round")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config.json
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    train_loader, test_loader = get_cifar10_dataloaders("./data", batch_size=config["batch_size"], use_fake=False)
    
    if config["partition_type"] == "iid":
        partitions, distributions = create_iid_partitions(train_loader.dataset, config["num_clients"], config["seed"])
    elif config["partition_type"] == "noniid":
        partitions, distributions = create_noniid_partitions(train_loader.dataset, config["num_clients"], config["dirichlet_alpha"], config["seed"])
        
    with open(os.path.join(args.output_dir, "client_class_histograms.json"), "w") as f:
        json.dump(distributions, f, indent=4)
        
    torch.save(partitions, os.path.join(args.output_dir, "client_partitions.pt"))
        
    asr_loader = create_asr_dataloader(test_loader, target_class=config["target_class"], trigger_size=config["trigger_size"], batch_size=config["batch_size"])
    untriggered_loader = create_untriggered_baseline_loader(test_loader, target_class=config["target_class"], batch_size=config["batch_size"])
    
    is_attack = (args.mode == "attack")
    print(f"=== Starting {args.mode.upper()} ===")
    results = run_fl_simulation(config, partitions, test_loader, asr_loader, untriggered_loader, is_attack, args.output_dir, args.log_trajectory)
    
    final_output = {
        "config": config,
        "mode": args.mode,
        "clean_accuracy": results["clean_eval"]["accuracy"],
        "clean_loss": results["clean_eval"]["loss"],
        "clean_macro_f1": results["clean_eval"]["macro_f1"],
        "triggered_rate": results["triggered_eval"]["accuracy"],
        "untriggered_airplane_rate": results["untriggered_airplane_rate"],
        "runtime_seconds": results["runtime_seconds"]
    }
    
    with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
        json.dump(final_output, f, indent=4)
        
    with open(os.path.join(args.output_dir, "runtime.json"), "w") as f:
        json.dump({"runtime_seconds": results["runtime_seconds"]}, f, indent=4)
        
    print(f"Finished {args.mode}. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
