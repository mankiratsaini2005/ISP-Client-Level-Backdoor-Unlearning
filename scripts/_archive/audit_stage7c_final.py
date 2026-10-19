import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset
import pandas as pd

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_, test_loader = get_cifar10_dataloaders("./data", 64, False)
test_dataset = test_loader.dataset

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

def get_kl_div(model_u, model_cf, loader):
    model_u.eval()
    model_cf.eval()
    total_kl = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out_u = model_u(x)
            out_cf = model_cf(x)
            kl = F.kl_div(
                F.log_softmax(out_u, dim=1),
                F.softmax(out_cf, dim=1),
                reduction='batchmean'
            )
            total_kl += kl.item() * x.size(0)
            total_samples += x.size(0)
    return total_kl / total_samples

def load_model(path):
    m = SmallCNN().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m

def get_latest_dir(prefix):
    runs = [d for d in os.listdir("results") if d.startswith(prefix)]
    if not runs: return None
    latest = max(runs, key=lambda d: os.path.getctime(os.path.join("results", d)))
    return os.path.join("results", latest)

conditions = {
    'D0': {'mcf': get_latest_dir("stage7b_c1_mcf_"), 'u_dir': "results/stage7b_unlearning/c1", "source": "results/stage7a_c1_attacked_20260910_233012"},
    'D1': {'mcf': get_latest_dir("stage7c_d1_mcf_"), 'u_dir': "results/stage7c_unlearning/d1", "source": get_latest_dir("stage7c_d1_attacked_")},
    'D2': {'mcf': get_latest_dir("stage7c_d2_mcf_"), 'u_dir': "results/stage7c_unlearning/d2", "source": get_latest_dir("stage7c_d2_attacked_")},
    'D3': {'mcf': get_latest_dir("stage7b_c3_mcf_"), 'u_dir': "results/stage7b_unlearning/c3", "source": "results/stage7a_c3_attacked_20260911_022619"}
}

results_df = []

for c, paths in conditions.items():
    if not paths['mcf'] or not os.path.exists(paths['u_dir']):
        print(f"Skipping {c}, missing artifacts")
        continue
        
    mcf_path = os.path.join(paths['mcf'], "checkpoints/global_round_100.pt")
    m_cf = load_model(mcf_path)
    cf_acc, cf_preds = evaluate(m_cf, eval_clean_loader)
    
    source_model_path = os.path.join(paths['source'], "checkpoints/global_round_100.pt")
    m_src = load_model(source_model_path)
    src_acc, _ = evaluate(m_src, eval_clean_loader)
    src_asr, _ = evaluate(m_src, eval_asr_loader)
    
    for m in ['subtraction', 'kd', 'federaser']:
        dir_path = os.path.join(paths['u_dir'], m)
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
        cf_gap = 100.0 - cf_agree
        
        kl = get_kl_div(m_u, m_cf, eval_clean_loader)
        
        results_df.append({
            'Condition': c,
            'Method': m,
            'Label Heterogeneity': "Low" if c in ["D0", "D2"] else "High",
            'Size Imbalance': "Low" if c in ["D0", "D1"] else "High",
            'Source Clean': src_acc,
            'Source ASR': src_asr,
            'MCF Clean': cf_acc,
            'Clean Acc': clean_acc,
            'Acc Diff': clean_acc - cf_acc,
            'ASR': asr,
            'CF Agreement': cf_agree,
            'CF Gap': cf_gap,
            'KL': kl
        })

df = pd.DataFrame(results_df)
os.makedirs("results/stage7c_audit", exist_ok=True)
df.to_csv("results/stage7c_audit/metric_recomputation.csv", index=False)
print(df.to_string())
