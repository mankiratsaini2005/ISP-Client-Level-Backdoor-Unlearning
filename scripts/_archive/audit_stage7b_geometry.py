import os
import torch
import numpy as np
import json

def load_updates(base_dir, round_idx):
    update_dir = os.path.join(base_dir, "client_updates", f"round_{round_idx:03d}")
    if not os.path.exists(update_dir):
        return None
        
    updates = {}
    for f in os.listdir(update_dir):
        if f.startswith("client_") and f.endswith(".pt"):
            cid = int(f.split('_')[1].split('.')[0])
            d = torch.load(os.path.join(update_dir, f), map_location='cpu', weights_only=False)
            vec = []
            for k,v in d['delta'].items():
                if v.is_floating_point() and 'running_' not in k and 'num_batches_tracked' not in k:
                    vec.append(v.flatten())
            if len(vec) == 0:
                continue
            updates[cid] = torch.cat(vec)
    return updates

dirs = {
    'C1_attacked': 'results/stage7a_c1_attacked_20260910_233012',
    'C2_attacked': 'results/stage7a_c2_attacked_20260911_005839',
    'C3_attacked': 'results/stage7a_c3_attacked_20260911_022619',
    'C4_attacked': 'results/stage7a_c4_attacked_20260911_035522'
}

results = []
for name, d in dirs.items():
    gamma = 10.0 if name == 'C4_attacked' else 5.0
    for r in [50, 75, 100]:
        updates = load_updates(d, r)
        if not updates or 0 not in updates:
            continue
            
        malicious_scaled = updates[0]
        benign = [v for k,v in updates.items() if k != 0]
        
        mal_norm = malicious_scaled.norm().item()
        raw_mal_norm = mal_norm / gamma
        
        benign_norms = [v.norm().item() for v in benign]
        avg_benign_norm = np.mean(benign_norms)
        
        benign_agg = torch.stack(benign).mean(dim=0)
        agg_norm = benign_agg.norm().item()
        
        cos_sim = torch.nn.functional.cosine_similarity(malicious_scaled, benign_agg, dim=0).item()
        
        dispersion = np.std(benign_norms) / avg_benign_norm
        
        results.append({
            'Condition': name,
            'Round': r,
            'Scaled_Mal_Norm': mal_norm,
            'Raw_Mal_Norm': raw_mal_norm,
            'Avg_Benign_Norm': avg_benign_norm,
            'Benign_Dispersion': dispersion,
            'Mal_Benign_Avg_Ratio': mal_norm / avg_benign_norm,
            'Cosine_Sim': cos_sim
        })

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("results/stage7a_audit/update_geometry_audit.csv", index=False)
print(df)
