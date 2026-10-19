import os
import time
import json
import torch
import torch.nn as nn
import argparse
import copy
import pandas as pd

from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.training.train_utils import evaluate
from src.data.backdoor import create_asr_dataloader

def verify_delta_scaling(save_dict, round_num):
    """
    Verifies that for rounds with scale_factor, the scaled delta is mathematically sound.
    """
    if "scale_factor" in save_dict and "raw_delta" in save_dict:
        scale_factor = save_dict["scale_factor"]
        raw_delta = save_dict["raw_delta"]
        scaled_delta = save_dict["delta"]
        
        # Verify scaling explicitly (e.g. gamma=5)
        if scale_factor != 5.0:
            print(f"  [WARNING] Round {round_num}: scale_factor is {scale_factor}, expected 5.0")
            
        # Verify tensor equality (within float tolerance)
        for k, v in scaled_delta.items():
            if v.is_floating_point():
                expected = raw_delta[k] * scale_factor
                diff = torch.max(torch.abs(v - expected)).item()
                if diff > 1e-5:
                    print(f"  [WARNING] Round {round_num}: Tensor {k} scale verification failed. Diff: {diff}")
    return

def perform_historical_subtraction(source_dir, output_dir, target_client, fedavg_weight):
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # 1. Determine participations
    sampling_path = os.path.join(source_dir, "client_sampling.json")
    with open(sampling_path, "r") as f:
        sampling = json.load(f)
        
    participations = []
    for r, clients in sampling.items():
        if target_client in clients:
            participations.append(int(r.split("_")[1]))
            
    participations.sort()
    
    pre_30 = [r for r in participations if r < 30]
    post_30 = [r for r in participations if r >= 30]
    
    print(f"Client {target_client} total participations: {len(participations)}")
    print(f"  Pre-round-30 (Clean)  : {len(pre_30)}")
    print(f"  Post-round-30 (Attack): {len(post_30)}")
    print(f"Participation rounds: {participations}")
    
    # 2. Load frozen Stage 3B global model
    global_path = os.path.join(source_dir, "checkpoints", "global_round_100.pt")
    print(f"Loading Stage 3B final model: {global_path}")
    M_3B = torch.load(global_path, weights_only=False)
    
    # Initialize accumulator
    accumulated_contributions = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in M_3B.items()}
    
    # 3. Process historical updates
    total_updates = 0
    total_elements = 0
    
    for round_num in participations:
        update_path = os.path.join(source_dir, "client_updates", f"round_{round_num:03d}", f"client_{target_client:02d}.pt")
        save_dict = torch.load(update_path, weights_only=False)
        
        # Verification tests
        verify_delta_scaling(save_dict, round_num)
        
        # Accumulate actual contribution
        delta = save_dict["delta"]
        for k in delta:
            # We enforce computation on float to avoid precision overflow
            contribution = (fedavg_weight * delta[k].float()).to(accumulated_contributions[k].device)
            accumulated_contributions[k] += contribution
            
            if total_updates == 0:
                total_elements += delta[k].numel()
                
        total_updates += 1
        
    # 4. Perform Model Reconstruction (Subtraction)
    M_5A = copy.deepcopy(M_3B)
    norm_sq = 0.0
    
    for k in M_3B:
        removed_contribution = accumulated_contributions[k]
        
        # Subtraction
        subtracted_val = M_3B[k].float() - removed_contribution
        
        if not M_3B[k].is_floating_point():
            subtracted_val = torch.round(subtracted_val).to(M_3B[k].dtype)
        else:
            if "running_var" in k:
                subtracted_val = torch.clamp(subtracted_val, min=1e-5)
            subtracted_val = subtracted_val.to(M_3B[k].dtype)
            norm_sq += torch.sum(removed_contribution ** 2).item()
            
        M_5A[k] = subtracted_val
        
    l2_norm = (norm_sq ** 0.5)
    
    # 5. Save Stage 5A Model
    out_path = os.path.join(output_dir, "checkpoints", "subtraction_model.pt")
    torch.save(M_5A, out_path)
    print(f"Saved Stage 5A unlearned model to {out_path}")
    
    # 6. Evaluate Model
    print("Evaluating unlearned model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SmallCNN().to(device)
    model.load_state_dict(M_5A)
    
    # Setup data
    _, test_loader = get_cifar10_dataloaders(batch_size=64)
    
    # Setup ASR test loader identical to Stage 3B
    asr_test_loader = create_asr_dataloader(
        test_loader,
        target_class=0,
        trigger_size=3,
        batch_size=64
    )
    
    criterion = nn.CrossEntropyLoss()
    
    eval_start_time = time.time()
    clean_loss, clean_acc = evaluate(model, test_loader, criterion, device)
    _, asr_acc = evaluate(model, asr_test_loader, criterion, device)
    eval_time = time.time() - eval_start_time
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 7. Save Metadata and Metrics
    metadata = {
        "source_dir": source_dir,
        "target_client": target_client,
        "fedavg_weight": fedavg_weight,
        "participation_rounds": participations,
        "total_participations": len(participations),
        "removed_l2_norm": l2_norm,
        "total_unlearning_time_s": total_time,
        "evaluation_time_s": eval_time,
        "additional_communication_bytes": 0,
        "clean_test_loss": clean_loss,
        "clean_test_acc": clean_acc,
        "asr": asr_acc
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    df = pd.DataFrame([metadata])
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        
    print("\n=== Stage 5A Subtraction & Evaluation Complete ===")
    print(f"Total Client Contributions Processed: {total_updates}")
    print(f"L2 Norm of Removed Contribution: {l2_norm:.4f}")
    print(f"Clean Test Loss: {clean_loss:.4f}")
    print(f"Clean Test Acc: {clean_acc:.2f}%")
    print(f"ASR: {asr_acc:.2f}%")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Additional Communication: 0 bytes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Path to Stage 3B results directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to Stage 5A output directory")
    parser.add_argument("--target_client", type=int, default=0, help="Client ID to unlearn")
    parser.add_argument("--fedavg_weight", type=float, default=0.1, help="FedAvg aggregation weight of the client")
    args = parser.parse_args()
    
    perform_historical_subtraction(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_client=args.target_client,
        fedavg_weight=args.fedavg_weight
    )
