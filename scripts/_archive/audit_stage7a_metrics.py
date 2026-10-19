import os
import json
import torch
from torch.utils.data import DataLoader, Subset
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_, test_loader = get_cifar10_dataloaders("./data", 64, False)
test_dataset = test_loader.dataset

# 1. Load canonical evaluation split
g = torch.Generator().manual_seed(42)
indices = torch.randperm(len(test_dataset), generator=g).tolist()
eval_indices = indices[2500:]

eval_clean_dataset = Subset(test_dataset, eval_indices)
eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)

eval_clean_loader = DataLoader(eval_clean_dataset, batch_size=64, shuffle=False)
eval_asr_loader = DataLoader(eval_asr_dataset, batch_size=64, shuffle=False)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    preds = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            preds.extend(pred.cpu().tolist())
    return 100.0 * correct / total, preds

def load_model(path):
    m = SmallCNN().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m

conditions = {
    'c1': 'results/stage7a_c1_mcf_20260911_003109',
    'c2': 'results/stage7a_c2_mcf_20260911_015906',
    'c3': 'results/stage7a_c3_mcf_20260911_032646',
    'c4': 'results/stage7a_c3_mcf_20260911_032646'
}

print("Regime,Method,Old_Clean,New_Clean,Clean_Diff,Old_ASR,New_ASR,ASR_Diff,Old_CF,New_CF,CF_Diff")

import pandas as pd
results_df = []

for c, mcf_dir in conditions.items():
    # 2. Checkpoint Selection Audit Check -> Use round 100 due to eval leakage
    mcf_path = os.path.join(mcf_dir, "checkpoints/global_round_100.pt")
        
    m_cf = load_model(mcf_path)
    _, cf_preds = evaluate(m_cf, eval_clean_loader)
    
    for m in ['subtraction', 'kd', 'federaser']:
        dir_path = os.path.join("results/stage7a_unlearning", c, m)
        if m == 'subtraction':
            model_path = os.path.join(dir_path, "checkpoints/subtraction_model.pt")
        elif m == 'kd':
            model_path = os.path.join(dir_path, "checkpoints/final_unlearned.pt")
        else:
            model_path = os.path.join(dir_path, "checkpoints/global_round_100.pt")
            
        if not os.path.exists(model_path):
            continue
            
        m_u = load_model(model_path)
        clean_acc, u_preds = evaluate(m_u, eval_clean_loader)
        asr, _ = evaluate(m_u, eval_asr_loader)
        
        agrees = sum(1 for a, b in zip(cf_preds, u_preds) if a == b)
        cf_agree = 100.0 * agrees / len(cf_preds)
        
        # Load OLD metrics from my previous flawed evaluation logic
        # Wait, the prompt implies "Do not trust the existing metrics.json values."
        # We don't have a metrics.json from my `evaluate_stage7a_unlearning.py` run, I only printed to console!
        # But wait, FedEraser and KD might have saved their own logs! Let's check metadata.json.
        old_clean = 0.0
        old_asr = 0.0
        old_cf = 0.0
        
        meta_path = os.path.join(dir_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                d = json.load(f)
                old_clean = d.get('clean_test_acc', d.get('clean_acc', 0.0))
                old_asr = d.get('asr', 0.0)
                old_cf = d.get('fidelity_agreement', d.get('fidelity', d.get('fidelity_to_reference', 0.0)))
        
        results_df.append({
            'Regime': c.upper(),
            'Method': m,
            'Old_Clean': old_clean,
            'New_Clean': clean_acc,
            'Clean_Diff': clean_acc - old_clean,
            'Old_ASR': old_asr,
            'New_ASR': asr,
            'ASR_Diff': asr - old_asr,
            'Old_CF': old_cf,
            'New_CF': cf_agree,
            'CF_Diff': cf_agree - old_cf
        })

df = pd.DataFrame(results_df)
os.makedirs("results/stage7a_audit", exist_ok=True)
df.to_csv("results/stage7a_audit/recomputed_metrics.csv", index=False)
print(df)
