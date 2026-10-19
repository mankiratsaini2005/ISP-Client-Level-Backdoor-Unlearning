import os
import json
import time
import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np
import math
from torch.utils.data import DataLoader, Subset

from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.training.train_utils import train_epoch, evaluate_extended
from src.data.fl_partition import create_iid_partitions, create_noniid_partitions
from src.data.backdoor import create_asr_dataloader

device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

target_client = 0
retained_rounds = list(range(1, 51))
lr = 0.01
momentum = 0.9
weight_decay = 0.0
batch_size = 64

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_l2_norm_of_floats(delta):
    squared_sum = 0.0
    for key, tensor in delta.items():
        if tensor.is_floating_point() and "running_" not in key and "num_batches_tracked" not in key:
            squared_sum += torch.sum(tensor ** 2).item()
    return math.sqrt(squared_sum)

def compute_model_delta(global_state, local_state):
    return {k: local_state[k] - global_state[k] for k in global_state.keys()}

def calibrate_update(historical_delta, calibration_delta):
    hist_norm = compute_l2_norm_of_floats(historical_delta)
    cali_norm = compute_l2_norm_of_floats(calibration_delta)
    
    calibrated_delta = {}
    scale_factor = 1.0
    if cali_norm > 1e-8:
        scale_factor = hist_norm / cali_norm
        
    for key in historical_delta.keys():
        tensor = calibration_delta[key]
        if tensor.is_floating_point() and "running_" not in key and "num_batches_tracked" not in key:
            calibrated_delta[key] = tensor * scale_factor
        else:
            calibrated_delta[key] = tensor.clone()
            
    if cali_norm > 1e-8:
        calibrated_norm = compute_l2_norm_of_floats(calibrated_delta)
        assert math.isclose(calibrated_norm, hist_norm, rel_tol=1e-4, abs_tol=1e-6), "Global norm mismatch"
            
    return calibrated_delta

def aggregate_fedavg_modified(global_state, client_deltas, sample_counts):
    total_samples = sum(sample_counts)
    if total_samples == 0: return {k: v.clone() for k, v in global_state.items()}
    weights = [n_k / total_samples for n_k in sample_counts]
    
    new_global_state = {k: v.clone() for k, v in global_state.items()}
    for key in global_state.keys():
        if "num_batches_tracked" in key:
            # COMPLETELY bypass delta aggregation for integer counters, preserving global value
            continue
            
        accumulated_delta = torch.zeros_like(global_state[key], dtype=torch.float32)
        for delta, weight in zip(client_deltas, weights):
            accumulated_delta += delta[key].to(accumulated_delta.device) * weight
            
        new_val = global_state[key].float() + accumulated_delta
        if not global_state[key].is_floating_point():
            new_val = new_val.round().to(global_state[key].dtype)
        new_global_state[key] = new_val
    return new_global_state

def run_federaser_setting(partition_type, cali_epochs, source_run, m_cf_path):
    t_setup_start = time.time()
    setting_name = f"{partition_type}_Ecal{cali_epochs}_v2"
    out_dir = f"results_phase2/federaser/{setting_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[{setting_name.upper()}] Setup...")
    train_loader_full, test_loader_full = get_cifar10_dataloaders(batch_size=batch_size, use_fake=False)
    test_dataset = test_loader_full.dataset
    
    # Check partitions exactly
    saved_partitions = torch.load(os.path.join(source_run, "client_partitions.pt"), weights_only=False)
    sp = saved_partitions if isinstance(saved_partitions, list) else saved_partitions.get("partitions", saved_partitions.get("client_partitions"))
    
    if partition_type == "iid":
        regen_parts, _ = create_iid_partitions(dataset=train_loader_full.dataset, num_clients=10, seed=42) # Uses dataset len 50000
    else:
        regen_parts, _ = create_noniid_partitions(dataset=train_loader_full.dataset, num_clients=10, alpha=0.5, seed=42)
        
    for c in range(10):
        regen_idx = sorted(regen_parts[c] if isinstance(regen_parts[c], list) else regen_parts[c].indices)
        save_idx = sorted(sp[c] if isinstance(sp[c], list) else sp[c].indices)
        assert regen_idx == save_idx, f"Partition mismatch for client {c}"
        print(f"[{setting_name}] Client {c} partitions match perfectly.")
    
    partitions = sp
    eval_clean_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    non_airplane_indices = [i for i, label in enumerate(test_dataset.targets) if label != 0]
    eval_non_airplane_dataset = Subset(test_dataset, non_airplane_indices)
    eval_asr_loader = create_asr_dataloader(test_loader_full, batch_size=batch_size, target_class=0, trigger_size=3)
    eval_untriggered_loader = DataLoader(eval_non_airplane_dataset, batch_size=batch_size, shuffle=False)
    airplane_indices = [i for i, label in enumerate(test_dataset.targets) if label == 0]
    eval_airplane_dataset = Subset(test_dataset, airplane_indices)
    eval_airplane_loader = DataLoader(eval_airplane_dataset, batch_size=batch_size, shuffle=False)

    global_model = SmallCNN().to(device)
    init_path = os.path.join(source_run, "initial_model.pt")
    global_model.load_state_dict(torch.load(init_path, map_location=device, weights_only=True))
    initial_counters = {k: v.item() for k, v in global_model.state_dict().items() if "num_batches_tracked" in k}
    
    criterion = nn.CrossEntropyLoss()
    client_ids = [c for c in range(10) if c != target_client]
    sample_counts = [len(partitions[c]) for c in client_ids]
    
    t_setup_end = time.time()
    setup_time = t_setup_end - t_setup_start
    
    print(f"[{setting_name.upper()}] Training Loop...")
    reconstruction_logs = []
    t_recon_start = time.time()
    
    for round_num in retained_rounds:
        round_start = time.time()
        client_calibrated_deltas = []
        
        for c in client_ids:
            update_path = os.path.join(source_run, "trajectory", f"round_{round_num:03d}", f"delta_client_{c:02d}.pt")
            hist_delta = torch.load(update_path, map_location=device, weights_only=False)
            if "delta" in hist_delta: hist_delta = hist_delta["delta"]
            
            if round_num == 1:
                tilde_delta = hist_delta
            else:
                seed_val = hash((42, round_num, c)) % (2**32)
                seed_everything(seed_val)
                g = torch.Generator()
                g.manual_seed(seed_val)
                
                client_loader = DataLoader(partitions[c], batch_size=batch_size, shuffle=True, generator=g)
                local_model = SmallCNN().to(device)
                local_model.load_state_dict(global_model.state_dict())
                
                optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                assert optimizer.defaults["weight_decay"] == 0.0, "Weight decay is not 0.0"
                
                for _ in range(cali_epochs):
                    train_epoch(local_model, client_loader, criterion, optimizer, device)
                
                cali_delta = compute_model_delta(global_model.state_dict(), local_model.state_dict())
                tilde_delta = calibrate_update(hist_delta, cali_delta)
                
            client_calibrated_deltas.append(tilde_delta)
            
        new_global_state = aggregate_fedavg_modified(global_model.state_dict(), client_calibrated_deltas, sample_counts)
        global_model.load_state_dict(new_global_state)
        
        # Assert counters unchanged
        for k, v in global_model.state_dict().items():
            if "num_batches_tracked" in k:
                assert v.item() == initial_counters[k], f"Counter {k} changed! {v.item()} != {initial_counters[k]}"
        
        round_time = time.time() - round_start
        reconstruction_logs.append({"round": round_num, "time_s": round_time})
        print(f"  Completed retained round {round_num:03d} in {round_time:.2f}s")
        
    t_recon_end = time.time()
    recon_time = t_recon_end - t_recon_start
    torch.save(global_model.state_dict(), os.path.join(out_dir, "final_model.pt"))
    
    print(f"[{setting_name.upper()}] Evaluating...")
    t_eval_start = time.time()
    clean_metrics = evaluate_extended(global_model, eval_clean_loader, criterion, device)
    asr_metrics = evaluate_extended(global_model, eval_asr_loader, criterion, device)
    
    utr_correct = 0
    utr_total = 0
    global_model.eval()
    with torch.no_grad():
        for x, _ in eval_untriggered_loader:
            x = x.to(device)
            p = global_model(x).argmax(1)
            utr_correct += (p == 0).sum().item()
            utr_total += x.size(0)
    utr = (utr_correct / utr_total) * 100.0 if utr_total > 0 else 0.0
    
    _, airplane_recall = evaluate_extended(global_model, eval_airplane_loader, criterion, device).get("accuracy", 0), evaluate_extended(global_model, eval_airplane_loader, criterion, device)["accuracy"]
    
    m_cf = SmallCNN().to(device)
    m_cf.load_state_dict(torch.load(m_cf_path, map_location=device, weights_only=True))
    m_cf.eval()
    
    def get_cf_agreement(loader):
        agree, total = 0, 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                p_fed = global_model(x).argmax(1)
                p_cf = m_cf(x).argmax(1)
                agree += (p_fed == p_cf).sum().item()
                total += x.size(0)
        return (agree / total) * 100.0 if total > 0 else 0.0

    cf_clean = get_cf_agreement(eval_clean_loader)
    cf_trig = get_cf_agreement(eval_asr_loader)
    t_eval_end = time.time()
    eval_time = t_eval_end - t_eval_start
    
    eval_results = {
        "Clean_Accuracy": clean_metrics["accuracy"],
        "Clean_Loss": clean_metrics["loss"],
        "Triggered_Rate": asr_metrics["accuracy"],
        "Untriggered_Airplane_Rate": utr,
        "Clean_CF_Agreement": cf_clean,
        "Triggered_CF_Agreement": cf_trig,
        "Airplane_Recall": airplane_recall,
        "Macro_F1": clean_metrics["macro_f1"] * 100.0
    }
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
        
    config = {
        "weight_decay": 0.0,
        "weight_decay_asserted": True,
        "norm_granularity": "global norm, archived implementation",
        "seed_scheme": "(42, round_num, client_id)",
        "backend_limitations": "cuDNN benchmarking may introduce slight non-determinism unless strictly disabled",
        "retained_rounds": retained_rounds,
        "cali_epochs": cali_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "momentum": momentum,
        "target_client": target_client,
        "setup_time_s": setup_time,
        "reconstruction_loop_time_s": recon_time,
        "reconstruction_loop_note": "Includes reading stored historical deltas from disk",
        "evaluation_time_s": eval_time
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    pd.DataFrame(reconstruction_logs).to_csv(os.path.join(out_dir, "reconstruction_logs.csv"), index=False)
    print(f"[{setting_name.upper()}] Done!")
    return eval_results

def run_controlled_timing():
    print("\n--- Controlled Timing Check ---")
    train_loader_full, test_loader_full = get_cifar10_dataloaders(batch_size=batch_size, use_fake=False)
    partitions, _ = create_iid_partitions(dataset=train_loader_full.dataset, num_clients=10, seed=42)
    criterion = nn.CrossEntropyLoss()
    
    # 1. 3 rounds M_CF-style training (9 clients, 2 local epochs)
    t1_start = time.time()
    global_model = SmallCNN().to(device)
    for r in range(3):
        for c in range(1, 10):
            local_model = SmallCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)
            client_loader = DataLoader(partitions[c], batch_size=batch_size, shuffle=True)
            for _ in range(2):
                train_epoch(local_model, client_loader, criterion, optimizer, device)
    t1_end = time.time()
    mcf_time = t1_end - t1_start
    mcf_per_epoch = mcf_time / (3 * 9 * 2)
    
    # 2. 3 rounds FedEraser Ecal=2 (retained rounds 2-4)
    t2_start = time.time()
    global_model = SmallCNN().to(device)
    source_run = "results_phase2/iid_attack_logged"
    io_time_acc = 0.0
    for r in range(2, 5):
        for c in range(1, 10):
            io_s = time.time()
            hist_delta = torch.load(os.path.join(source_run, "trajectory", f"round_{r:03d}", f"delta_client_{c:02d}.pt"), map_location=device, weights_only=False)
            if "delta" in hist_delta: hist_delta = hist_delta["delta"]
            io_time_acc += (time.time() - io_s)
            
            local_model = SmallCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0)
            client_loader = DataLoader(partitions[c], batch_size=batch_size, shuffle=True)
            for _ in range(2):
                train_epoch(local_model, client_loader, criterion, optimizer, device)
            cali_delta = compute_model_delta(global_model.state_dict(), local_model.state_dict())
            _ = calibrate_update(hist_delta, cali_delta)
    t2_end = time.time()
    fe_time = t2_end - t2_start
    fe_per_epoch = fe_time / (3 * 9 * 2)
    
    res = {
        "m_cf_total_3_rounds_s": mcf_time,
        "m_cf_per_client_epoch_s": mcf_per_epoch,
        "fe_total_3_rounds_s": fe_time,
        "fe_per_client_epoch_s": fe_per_epoch,
        "fe_io_overhead_s": io_time_acc,
        "fe_compute_only_s": fe_time - io_time_acc,
        "note": "Includes purely training logic across 9 clients * 3 rounds * 2 epochs. FedEraser timing includes historical disk reading (isolated via fe_io_overhead_s). No logging/eval included."
    }
    with open("results_phase2/federaser/controlled_timing.json", "w") as f:
        json.dump(res, f, indent=4)
    print("Controlled timing completed.")

if __name__ == "__main__":
    run_federaser_setting("iid", 1, "results_phase2/iid_attack_logged", "results_phase2/iid_counterfactual/final_model.pt")
    run_federaser_setting("iid", 2, "results_phase2/iid_attack_logged", "results_phase2/iid_counterfactual/final_model.pt")
    run_federaser_setting("noniid", 1, "results_phase2/noniid_attack_logged", "results_phase2/noniid_counterfactual/final_model.pt")
    run_federaser_setting("noniid", 2, "results_phase2/noniid_attack_logged", "results_phase2/noniid_counterfactual/final_model.pt")
    run_controlled_timing()
