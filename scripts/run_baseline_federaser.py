import os
import json
import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Subset

from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.training.fedavg import compute_model_delta, aggregate_fedavg
from src.training.train_utils import train_epoch, evaluate
from src.unlearning.federaser import calibrate_update
from src.data.fl_partition import create_iid_partitions, create_noniid_partitions
from src.data.backdoor import create_asr_dataloader

device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Fixed Settings
target_client = 0
retained_rounds = list(range(1, 50, 2))  # 1, 3, 5, ..., 49
lr = 0.01
momentum = 0.9
batch_size = 64
cali_epochs = 1

def run_federaser_setting(setting_name, source_run, m_cf_path):
    out_dir = f"results_phase2/federaser/{setting_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[{setting_name.upper()}] Starting FedEraser Baseline...")
    start_time = time.time()
    
    # 1. Load Data
    train_loader_full, test_loader_full = get_cifar10_dataloaders(batch_size=batch_size, use_fake=False)
    
    if setting_name == "iid":
        partitions, _ = create_iid_partitions(train_loader_full.dataset, num_clients=10, seed=42)
    else:
        partitions, _ = create_noniid_partitions(train_loader_full.dataset, num_clients=10, alpha=0.5, seed=42)
        
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(test_loader_full.dataset), generator=g).tolist()
    eval_indices = indices[2500:]
    eval_clean_dataset = Subset(test_loader_full.dataset, eval_indices)
    eval_clean_loader = DataLoader(eval_clean_dataset, batch_size=batch_size, shuffle=False)
    
    # Using robust ASR dataloader directly from source
    eval_asr_loader = create_asr_dataloader(test_loader_full, batch_size=batch_size, target_class=0, trigger_size=3)
    
    airplane_indices = [idx for idx in eval_indices if test_loader_full.dataset.targets[idx] == 0]
    eval_airplane_dataset = Subset(test_loader_full.dataset, airplane_indices)
    eval_airplane_loader = DataLoader(eval_airplane_dataset, batch_size=batch_size, shuffle=False)

    global_model = SmallCNN().to(device)
    init_path = os.path.join(source_run, "initial_model.pt")
    global_model.load_state_dict(torch.load(init_path, map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    total_cali_epochs = 0
    reconstruction_logs = []
    
    client_ids = [c for c in range(10) if c != target_client]
    sample_counts = [len(partitions[c]) for c in client_ids]
    
    for round_num in retained_rounds:
        round_start = time.time()
        client_calibrated_deltas = []
        
        for c in client_ids:
            update_path = os.path.join(source_run, "trajectory", f"round_{round_num:03d}", f"delta_client_{c:02d}.pt")
            hist_delta = torch.load(update_path, map_location=device)
            if "delta" in hist_delta: hist_delta = hist_delta["delta"]
            
            if round_num == 1:
                tilde_delta = hist_delta
            else:
                client_loader = DataLoader(partitions[c], batch_size=batch_size, shuffle=True)
                local_model = SmallCNN().to(device)
                local_model.load_state_dict(global_model.state_dict())
                
                optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
                
                for _ in range(cali_epochs):
                    train_epoch(local_model, client_loader, criterion, optimizer, device)
                total_cali_epochs += cali_epochs
                
                cali_delta = compute_model_delta(global_model.state_dict(), local_model.state_dict())
                tilde_delta = calibrate_update(hist_delta, cali_delta)
                
            client_calibrated_deltas.append(tilde_delta)
            
        new_global_state = aggregate_fedavg(global_model.state_dict(), client_calibrated_deltas, sample_counts)
        global_model.load_state_dict(new_global_state)
        
        round_time = time.time() - round_start
        reconstruction_logs.append({"round": round_num, "time_s": round_time})
        print(f"  Completed retained round {round_num:03d} in {round_time:.2f}s")
        
    total_time = time.time() - start_time
    torch.save(global_model.state_dict(), os.path.join(out_dir, "final_model.pt"))
    
    print(f"[{setting_name.upper()}] Evaluating...")
    clean_loss, clean_acc = evaluate(global_model, eval_clean_loader, criterion, device)
    _, asr = evaluate(global_model, eval_asr_loader, criterion, device)
    _, utr = evaluate(global_model, eval_airplane_loader, criterion, device)
    
    m_cf = SmallCNN().to(device)
    m_cf.load_state_dict(torch.load(m_cf_path, map_location=device))
    m_cf.eval()
    global_model.eval()
    
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
    
    correct = {i: 0 for i in range(10)}
    total_c = {i: 0 for i in range(10)}
    with torch.no_grad():
        for x, y in test_loader_full:
            x, y = x.to(device), y.to(device)
            p = global_model(x).argmax(1)
            for pred, true_label in zip(p, y):
                if pred.item() == true_label.item():
                    correct[true_label.item()] += 1
                total_c[true_label.item()] += 1
    
    per_class_acc = {i: (correct[i]/total_c[i])*100.0 for i in range(10)}
    macro_f1 = sum(per_class_acc.values()) / 10.0
    
    eval_results = {
        "Clean_Accuracy": clean_acc,
        "Clean_Loss": clean_loss,
        "Triggered_Rate": asr,
        "Untriggered_Airplane_Rate": utr,
        "Clean_CF_Agreement": cf_clean,
        "Triggered_CF_Agreement": cf_trig,
        "Airplane_Recall": per_class_acc[0],
        "Macro_F1": macro_f1,
        "Per_Class_Clean_Accuracy": per_class_acc
    }
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
        
    config = {
        "retained_rounds": retained_rounds,
        "cali_epochs": cali_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "momentum": momentum,
        "target_client": target_client,
        "total_calibration_epochs": total_cali_epochs,
        "reconstruction_time_seconds": total_time
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    pd.DataFrame(reconstruction_logs).to_csv(os.path.join(out_dir, "reconstruction_logs.csv"), index=False)
    print(f"[{setting_name.upper()}] Done in {total_time:.1f}s")
    return eval_results, total_time

if __name__ == "__main__":
    iid_res, iid_time = run_federaser_setting("iid", "results_phase2/iid_attack_logged", "results_phase2/iid_counterfactual/final_model.pt")
    niid_res, niid_time = run_federaser_setting("noniid", "results_phase2/noniid_attack_logged", "results_phase2/noniid_counterfactual/final_model.pt")
