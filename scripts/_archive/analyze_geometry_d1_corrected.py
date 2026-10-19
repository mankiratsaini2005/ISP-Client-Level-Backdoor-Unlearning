import torch
import json
import os
import math
from src.models.small_cnn import SmallCNN

run_dir = "results/stage7c_d1_attacked_20260911_230919"
rounds = [24, 49, 73, 100]

def get_trainable_params(state_dict):
    return {k: v for k, v in state_dict.items() if v.is_floating_point() and 'num_batches_tracked' not in k}

def flatten_dict(d):
    return torch.cat([v.flatten() for v in d.values()])

print("Analyzing Update Geometry for D1 (Corrected, no double-scaling)...")
print("-" * 50)

for r in rounds:
    print(f"\n--- Round {r} ---")
    round_dir = os.path.join(run_dir, "client_updates", f"round_{r:03d}")
    
    # Check if client 0 is there
    c0_path = os.path.join(round_dir, "client_00.pt")
    if not os.path.exists(c0_path):
        print("Client 0 not found.")
        continue
    
    # Load client 0 update
    c0_data = torch.load(c0_path, map_location="cpu")
    c0_delta = get_trainable_params(c0_data["delta"])
    c0_flat = flatten_dict(c0_delta)
    
    mal_norm = torch.norm(c0_flat).item()
    print(f"Malicious Actual Aggregation Norm: {mal_norm:.4f}")
    
    # Load benign updates
    benign_flats = []
    for f in os.listdir(round_dir):
        if f != "client_00.pt" and f.endswith(".pt"):
            b_data = torch.load(os.path.join(round_dir, f), map_location="cpu")
            b_delta = get_trainable_params(b_data["delta"])
            b_flat = flatten_dict(b_delta)
            benign_flats.append(b_flat)
            
    b_norms = [torch.norm(bf).item() for bf in benign_flats]
    mean_b_norm = sum(b_norms) / len(b_norms)
    std_b_norm = torch.tensor(b_norms).std().item()
    cv_b_norm = std_b_norm / mean_b_norm if mean_b_norm > 0 else 0
    
    print(f"Mean Benign Norm: {mean_b_norm:.4f}")
    print(f"Malicious/Benign Norm Ratio: {mal_norm / mean_b_norm:.2f}x")
    print(f"Benign CV (Norm Dispersion): {cv_b_norm:.4f}")
    
    # Directional dispersion (cosine sim)
    cos_sims = []
    for i in range(len(benign_flats)):
        for j in range(i + 1, len(benign_flats)):
            sim = torch.nn.functional.cosine_similarity(benign_flats[i].unsqueeze(0), benign_flats[j].unsqueeze(0)).item()
            cos_sims.append(sim)
    
    mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0
    dispersion = 1 - mean_cos
    print(f"Benign Directional Dispersion (1 - cos): {dispersion:.4f}")
    
    # Malicious vs Benign mean cosine
    mb_sims = []
    for bf in benign_flats:
        sim = torch.nn.functional.cosine_similarity(c0_flat.unsqueeze(0), bf.unsqueeze(0)).item()
        mb_sims.append(sim)
        
    mean_mb_cos = sum(mb_sims) / len(mb_sims)
    print(f"Malicious-to-Benign Mean Cosine: {mean_mb_cos:.4f}")

