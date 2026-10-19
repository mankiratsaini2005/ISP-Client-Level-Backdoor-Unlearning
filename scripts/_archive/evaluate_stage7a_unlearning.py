import os
import torch
import torch.nn as nn
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_, test_loader = get_cifar10_dataloaders("./data", 64, False)
asr_loader = create_asr_dataloader(test_loader, target_class=0, trigger_size=3, batch_size=64)

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

print("| Regime | Method | Clean Acc | ASR | CF Agreement | CF Gap | Runtime |")
print("|---|---|---:|---:|---:|---:|---:|")

for c, mcf_dir in conditions.items():
    mcf_path = os.path.join(mcf_dir, "checkpoints/best_global.pt")
    if not os.path.exists(mcf_path):
        mcf_path = os.path.join(mcf_dir, "checkpoints/global_round_100.pt")
        
    m_cf = load_model(mcf_path)
    _, cf_preds = evaluate(m_cf, test_loader)
    
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
        clean_acc, u_preds = evaluate(m_u, test_loader)
        asr, _ = evaluate(m_u, asr_loader)
        
        agrees = sum(1 for a, b in zip(cf_preds, u_preds) if a == b)
        cf_agree = 100.0 * agrees / len(cf_preds)
        cf_gap = 100.0 - cf_agree
        
        import json
        meta_path = os.path.join(dir_path, "metadata.json")
        runtime = 0.0
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                d = json.load(f)
                runtime = d.get('total_unlearning_time_s', d.get('total_kd_time_s', d.get('total_time_s', 0.0)))
                
        c_upper = c.upper()
        m_format = "Subtraction" if m == "subtraction" else ("Subtraction+KD" if m == "kd" else "FedEraser")
        print(f"| {c_upper} | {m_format} | {clean_acc:.2f}% | {asr:.2f}% | {cf_agree:.2f}% | {cf_gap:.2f}% | {runtime:.1f}s |")
