import os
import torch
import numpy as np

def load_updates(base_dir, round_idx):
    update_dir = os.path.join(base_dir, "client_updates", f"round_{round_idx:03d}")
    if not os.path.exists(update_dir):
        return None
        
    updates = {}
    for f in os.listdir(update_dir):
        if f.startswith("client_") and f.endswith(".pt"):
            cid = int(f.split('_')[1].split('.')[0])
            d = torch.load(os.path.join(update_dir, f), map_location='cpu')
            # flatten delta
            vec = torch.cat([v.flatten() for v in d['delta'].values()])
            updates[cid] = vec
    return updates

dirs = {
    'C1 (H0)': 'results/stage7a_c1_attacked_20260910_233012',
    'C3 (H2)': 'results/stage7a_c3_attacked_20260911_022619',
    'C4 (H2+A1)': 'results/stage7a_c4_attacked_20260911_035522'
}

print("=== UPDATE GEOMETRY (Round 100) ===")
for name, d in dirs.items():
    updates = load_updates(d, 100)
    if not updates or 0 not in updates:
        continue
        
    malicious = updates[0]
    benign = [v for k,v in updates.items() if k != 0]
    
    mal_norm = malicious.norm().item()
    benign_norms = [v.norm().item() for v in benign]
    avg_benign_norm = np.mean(benign_norms)
    
    benign_agg = torch.stack(benign).mean(dim=0)
    agg_norm = benign_agg.norm().item()
    
    cos_sim = torch.nn.functional.cosine_similarity(malicious, benign_agg, dim=0).item()
    
    dispersion = np.std(benign_norms) / avg_benign_norm
    
    print(f"\n{name}:")
    print(f"  Malicious Norm       : {mal_norm:.4f}")
    print(f"  Avg Benign Norm      : {avg_benign_norm:.4f}")
    print(f"  Malicious/Benign Avg : {mal_norm / avg_benign_norm:.2f}x")
    print(f"  Benign Dispersion    : {dispersion:.4f}")
    print(f"  Cosine Sim (Mal, Ben): {cos_sim:.4f}")
