import torch
import torch.nn.functional as F
import json
import os
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_, test_loader = get_cifar10_dataloaders("./data", 64, False)
test_dataset = test_loader.dataset

g = torch.Generator().manual_seed(42)
indices = torch.randperm(len(test_dataset), generator=g).tolist()
eval_indices = indices[2500:]

eval_clean_dataset = torch.utils.data.Subset(test_dataset, eval_indices)
eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)

eval_clean_loader = torch.utils.data.DataLoader(eval_clean_dataset, batch_size=64, shuffle=False)
eval_asr_loader = torch.utils.data.DataLoader(eval_asr_dataset, batch_size=64, shuffle=False)

def evaluate(model, ref_model=None):
    model.eval()
    if ref_model: ref_model.eval()
    
    correct = 0
    total = 0
    asr_correct = 0
    asr_total = 0
    agree = 0
    kl_sum = 0
    
    with torch.no_grad():
        for x, y in eval_clean_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            if ref_model:
                out_ref = ref_model(x)
                pred_ref = out_ref.argmax(dim=1)
                agree += (pred == pred_ref).sum().item()
                
                log_p = F.log_softmax(out, dim=1)
                q = F.softmax(out_ref, dim=1)
                kl_sum += F.kl_div(log_p, q, reduction='sum').item()
                
        for x, y in eval_asr_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            asr_correct += (pred == y).sum().item()
            asr_total += y.size(0)
            
    res = {
        "Clean Accuracy": 100.0 * correct / total,
        "ASR": 100.0 * asr_correct / asr_total if asr_total > 0 else 0
    }
    if ref_model:
        res["CF Agreement"] = 100.0 * agree / total
        res["KL Divergence"] = kl_sum / total
    return res

models = {
    "Attacked": "results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423/checkpoints/global_round_100.pt",
    "M_CF": "results/stage7c_d2_mcf20260911_213425/checkpoints/global_round_100.pt",
    "Subtraction": "results/stage7c_d2/unlearning/subtraction/checkpoints/subtraction_model.pt",
    "KD": "results/stage7c_d2/unlearning/kd/checkpoints/student_model.pt",
    "FedEraser": "results/stage7c_d2/unlearning/federaser/checkpoints/federaser_round_100.pt"
}

m_cf = SmallCNN().to(device)
m_cf.load_state_dict(torch.load(models["M_CF"], map_location=device))
ref_acc = evaluate(m_cf)["Clean Accuracy"]

results = {}
for name, path in models.items():
    m = SmallCNN().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    if name == "M_CF":
        res = evaluate(m)
        res["CF Agreement"] = 100.0
        res["KL Divergence"] = 0.0
        res["CF Gap"] = 0.0
    else:
        res = evaluate(m, ref_model=m_cf)
        res["CF Gap"] = ref_acc - res["Clean Accuracy"]
    results[name] = res

print(json.dumps(results, indent=4))
