import json
import torch
import torch.nn as nn
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader
from torch.utils.data import DataLoader

device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

batch_size = 64
_, test_loader_full = get_cifar10_dataloaders(batch_size=batch_size, use_fake=False)

eval_clean_loader = DataLoader(test_loader_full.dataset, batch_size=batch_size, shuffle=False)
eval_asr_loader = create_asr_dataloader(test_loader_full, batch_size=batch_size, target_class=0, trigger_size=3)

def get_cf_agreement(m_main_path, m_cf_path):
    m_main = SmallCNN().to(device)
    m_main.load_state_dict(torch.load(m_main_path, map_location=device, weights_only=True))
    m_main.eval()
    
    m_cf = SmallCNN().to(device)
    m_cf.load_state_dict(torch.load(m_cf_path, map_location=device, weights_only=True))
    m_cf.eval()
    
    def agree(loader):
        ag, tot = 0, 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                p1 = m_main(x).argmax(1)
                p2 = m_cf(x).argmax(1)
                ag += (p1 == p2).sum().item()
                tot += x.size(0)
        return (ag / tot) * 100.0 if tot > 0 else 0.0

    return agree(eval_clean_loader), agree(eval_asr_loader)

iid_clean, iid_trig = get_cf_agreement("results/iid_clean/final_model.pt", "results_phase2/iid_counterfactual/final_model.pt")
niid_clean, niid_trig = get_cf_agreement("results/noniid_clean/final_model.pt", "results_phase2/noniid_counterfactual/final_model.pt")

with open("results/iid_clean/cf_agreement.json", "w") as f:
    json.dump({"Clean_CF_Agreement": iid_clean, "Triggered_CF_Agreement": iid_trig}, f, indent=4)
    
with open("results/noniid_clean/cf_agreement.json", "w") as f:
    json.dump({"Clean_CF_Agreement": niid_clean, "Triggered_CF_Agreement": niid_trig}, f, indent=4)

print(f"IID Phase-1 Clean vs M_CF -> Clean: {iid_clean:.2f}%, Triggered: {iid_trig:.2f}%")
print(f"Non-IID Phase-1 Clean vs M_CF -> Clean: {niid_clean:.2f}%, Triggered: {niid_trig:.2f}%")
