import os
import json
import time
import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader
from src.training.fedavg import compute_model_delta, aggregate_fedavg
from src.training.train_utils import train_epoch, evaluate
from src.unlearning.federaser import calibrate_update

def run_federaser(source_run, reference_path, out_dir, smoke_test=False):
    start_time = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Paths and Configs
    target_client = 0
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    
    # 2. Datasets
    print("Loading datasets...")
    train_loader_full, test_loader_full = get_cifar10_dataloaders(batch_size=64)
    train_dataset = train_loader_full.dataset
    test_dataset = test_loader_full.dataset
    
    from src.data.fl_partition import create_iid_partitions
    partitions, _ = create_iid_partitions(train_dataset, num_clients=20, seed=42)
    
    # Deterministic subset for fair comparison
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(test_dataset), generator=g).tolist()
    eval_indices = indices[2500:]
    
    # Save indices to prove consistency
    with open(os.path.join(out_dir, "eval_indices.json"), "w") as f:
        json.dump(eval_indices, f)
    
    eval_clean_dataset = Subset(test_dataset, eval_indices)
    eval_clean_loader = DataLoader(eval_clean_dataset, batch_size=64, shuffle=False)
    
    # 3. Model Initialization
    global_model = SmallCNN().to(device)
    # Start from round 0 (fresh initialization from canonical trajectory)
    global_model.load_state_dict(torch.load(os.path.join(source_run, "checkpoints", "global_round_000.pt"), weights_only=False))
    
    # 4. Load Participation History
    with open(os.path.join(source_run, "client_sampling.json"), "r") as f:
        sampling_dict = json.load(f)
    # Keys are string "1", "2", ..., "100"
    
    max_rounds = 2 if smoke_test else 100
    
    criterion = nn.CrossEntropyLoss()
    total_cali_epochs = 0
    total_new_comm_bytes = 0
    
    import csv
    metrics_file = os.path.join(out_dir, "metrics.csv")
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "retained_clients", "clean_acc", "asr", "elapsed_unlearn_time_s", "cum_comm_bytes"])
        
    print(f"Starting FedEraser reconstruction for {max_rounds} rounds...")
    
    unlearn_time_total = 0.0
    
    # Pre-create ASR dataloader for round-by-round eval
    from scripts.unlearn_stage5b import create_filtered_asr_dataset
    eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)
    eval_asr_loader = DataLoader(eval_asr_dataset, batch_size=64, shuffle=False)
    
    for round_num in range(1, max_rounds + 1):
        round_start_time = time.time()
        
        round_str = f"round_{round_num:03d}"
        if round_str not in sampling_dict:
            break
            
        sampled_clients = sampling_dict[round_str]
        
        # Target Client Exclusion!
        retained_clients = [c for c in sampled_clients if c != target_client]
        
        client_calibrated_deltas = []
        client_sample_counts = []
        
        for client_id in retained_clients:
            # A. Load Historical Update U_k
            update_path = os.path.join(source_run, "client_updates", f"round_{round_num:03d}", f"client_{client_id:02d}.pt")
            if not os.path.exists(update_path):
                raise FileNotFoundError(f"Historical update missing: {update_path}")
                
            historical_data = torch.load(update_path, weights_only=False)
            historical_delta = historical_data["delta"]
            historical_delta = {k: v.to(device) for k, v in historical_delta.items()}
            
            # B. Calibration Training to get \hat{U}_k
            client_subset = partitions[client_id]
            client_loader = DataLoader(client_subset, batch_size=64, shuffle=True)
            
            local_model = SmallCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            for _ in range(2):
                train_epoch(local_model, client_loader, criterion, optimizer, device)
            total_cali_epochs += 2
            
            total_new_comm_bytes += 2 * sum(p.numel() * 4 for p in local_model.parameters())
            
            new_calibration_delta = compute_model_delta(global_model.state_dict(), local_model.state_dict())
            new_calibration_delta = {k: v.to(device) for k, v in new_calibration_delta.items()}
            
            # C. Update Calibration
            tilde_delta = calibrate_update(historical_delta, new_calibration_delta)
            
            client_calibrated_deltas.append(tilde_delta)
            client_sample_counts.append(len(partitions[client_id]))
            
        # D. Aggregation
        new_global_state = aggregate_fedavg(global_model.state_dict(), client_calibrated_deltas, client_sample_counts)
        global_model.load_state_dict(new_global_state)
        
        unlearn_time_total += (time.time() - round_start_time)
        
        # Eval Phase (not counted in unlearn time)
        _, test_acc = evaluate(global_model, eval_clean_loader, criterion, device)
        _, test_asr = evaluate(global_model, eval_asr_loader, criterion, device)
        
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round_num, len(retained_clients), test_acc, test_asr, unlearn_time_total, total_new_comm_bytes])
            
        print(f"Round {round_num:03d} | Retained: {len(retained_clients)} | Acc: {test_acc:.2f}% | ASR: {test_asr:.2f}% | Time: {unlearn_time_total:.1f}s | Comm: {total_new_comm_bytes / 1e6:.1f}MB")
        
    # --- EVALUATION ---
    print("Evaluating final reconstructed model fidelity...")
    test_loss, test_acc = evaluate(global_model, eval_clean_loader, criterion, device)
    _, test_asr = evaluate(global_model, eval_asr_loader, criterion, device)
    
    # Counterfactual Fidelity Eval
    # We compute prediction agreement against the true Matched Counterfactual on the clean eval set
    ref_model = SmallCNN().to(device)
    ref_model.load_state_dict(torch.load(reference_path, weights_only=False))
    ref_model.eval()
    global_model.eval()
    
    agree_count = 0
    total_count = 0
    with torch.no_grad():
        for inputs, _ in eval_clean_loader:
            inputs = inputs.to(device)
            out_fed = global_model(inputs)
            out_ref = ref_model(inputs)
            
            pred_fed = out_fed.max(1)[1]
            pred_ref = out_ref.max(1)[1]
            
            agree_count += (pred_fed == pred_ref).sum().item()
            total_count += inputs.size(0)
            
    agreement = 100.0 * agree_count / total_count
    
    total_time = time.time() - start_time
    
    print(f"--- FedEraser {'SMOKE ' if smoke_test else ''}Results ---")
    print(f"Clean Acc: {test_acc:.2f}%")
    print(f"ASR:       {test_asr:.2f}%")
    print(f"Fidelity:  {agreement:.2f}% agreement with Stage 4 at round {max_rounds}")
    print(f"Runtime:   {total_time:.2f} s")
    print(f"Total Cal Epochs: {total_cali_epochs}")
    print(f"New Comm Bytes: {total_new_comm_bytes}")
    
    # Save checkpoint
    torch.save(global_model.state_dict(), os.path.join(out_dir, "checkpoints", f"global_round_{max_rounds:03d}.pt"))
    
    # Save metadata
    metadata = {
        "max_rounds": max_rounds,
        "smoke_test": smoke_test,
        "clean_acc": test_acc,
        "asr": test_asr,
        "fidelity_agreement": agreement,
        "total_time_s": total_time,
        "total_calibration_epochs": total_cali_epochs,
        "new_communication_bytes": total_new_comm_bytes,
        "adaptations": [
            "Aggregation weights renormalized over retained clients (e.g. 1/9).",
            "BatchNorm/Integer buffers bypass L2 magnitude scaling."
        ]
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Artifacts saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing compromised run")
    parser.add_argument("--reference_path", type=str, required=True, help="Path to Matched Counterfactual checkpoint")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run only 2 rounds for validation")
    args = parser.parse_args()
    
    run_federaser(args.source_dir, args.reference_path, args.out_dir, smoke_test=args.smoke_test)
