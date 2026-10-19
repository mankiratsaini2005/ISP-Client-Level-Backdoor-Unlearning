import torch
import os
import json
import numpy as np

def get_update_vector(path):
    data = torch.load(path, map_location="cpu")
    sd = data["delta"] if "delta" in data else data
    vec = []
    for k, v in sd.items():
        if "num_batches_tracked" in k or "running_mean" in k or "running_var" in k:
            continue
        if not isinstance(v, torch.Tensor) or not v.is_floating_point():
            continue
        vec.append(v.flatten())
    if not vec:
        raise ValueError(f"No valid tensors in {path}")
    return torch.cat(vec)

def analyze_round(base_dir, round_num):
    updates_dir = os.path.join(base_dir, f"client_updates/round_{round_num:03d}")
    with open(os.path.join(base_dir, "client_sampling.json")) as f:
        sampling = json.load(f)
        
    clients = sampling[f"round_{round_num:03d}"]
    if 0 not in clients:
        return None
        
    vectors = {}
    for c in clients:
        path = os.path.join(updates_dir, f"client_{c:02d}.pt")
        vectors[c] = get_update_vector(path)
        
    malicious_scaled_vec = vectors[0] # Stored update IS the scaled update
    benign_vecs = [vectors[c] for c in clients if c != 0]
    
    malicious_scaled_norm = torch.norm(malicious_scaled_vec).item()
    malicious_raw_norm = malicious_scaled_norm / 5.0 # since we know gamma=5.0
    
    benign_norms = [torch.norm(v).item() for v in benign_vecs]
    mean_benign_norm = np.mean(benign_norms)
    std_benign_norm = np.std(benign_norms)
    cv_benign = std_benign_norm / mean_benign_norm if mean_benign_norm > 0 else 0
    
    malicious_ratio = malicious_scaled_norm / mean_benign_norm if mean_benign_norm > 0 else 0
    
    # Cosine similarities
    stacked_benign = torch.stack(benign_vecs)
    benign_mean_vec = torch.mean(stacked_benign, dim=0)
    
    cos = torch.nn.CosineSimilarity(dim=0)
    mal_vs_benign_cos = cos(malicious_scaled_vec, benign_mean_vec).item()
    
    # Pairwise benign cos
    pairwise_cos = []
    for i in range(len(benign_vecs)):
        for j in range(i+1, len(benign_vecs)):
            pairwise_cos.append(cos(benign_vecs[i], benign_vecs[j]).item())
    
    mean_pairwise_cos = np.mean(pairwise_cos)
    benign_dispersion = 1.0 - mean_pairwise_cos
    
    return {
        "actual_round": round_num,
        "malicious_raw_norm": malicious_raw_norm,
        "malicious_actual_aggregation_norm": malicious_scaled_norm,
        "mean_benign_norm": mean_benign_norm,
        "std_benign_norm": std_benign_norm,
        "cv_benign_norm": cv_benign,
        "malicious_ratio": malicious_ratio,
        "malicious_vs_benign_cos": mal_vs_benign_cos,
        "mean_pairwise_benign_cos": mean_pairwise_cos,
        "benign_dispersion": benign_dispersion
    }

base = "results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423"
res = []
for r in [25, 50, 75, 100]:
    for offset in range(10):
        data = analyze_round(base, r - offset)
        if data:
            res.append(data)
            break

with open("results/stage7c_d2/geometry_analysis_corrected.json", "w") as f:
    json.dump(res, f, indent=4)
print(json.dumps(res, indent=4))
