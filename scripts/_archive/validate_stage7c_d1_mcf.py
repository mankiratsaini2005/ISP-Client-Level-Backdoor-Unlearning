import torch
import json
import os
import math
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset

attacked_dir = "results/stage7c_d1_attacked_20260911_230919"
mcf_dir = "results/stage7c_d1_mcf_20260911_230957"

print("Validating M_CF Structural Integrity...")
# Check 100 rounds
assert os.path.exists(os.path.join(mcf_dir, "checkpoints", "global_round_100.pt")), "M_CF did not reach round 100"
assert os.path.exists(os.path.join(attacked_dir, "checkpoints", "global_round_100.pt")), "Attacked did not reach round 100"

# Check sampling schedule
with open(os.path.join(attacked_dir, "client_sampling.json"), "r") as f:
    attack_sampling = json.load(f)
with open(os.path.join(mcf_dir, "client_sampling.json"), "r") as f:
    mcf_sampling = json.load(f)

for r in range(1, 101):
    k = f"round_{r:03d}"
    atk_clients = attack_sampling[k]
    mcf_clients = mcf_sampling[k]
    
    # M_CF should be exact same except Client 0 is removed, and NO replacement
    expected = [c for c in atk_clients if c != 0]
    assert mcf_clients == expected, f"Round {r} sampling mismatch. Atk: {atk_clients}, M_CF: {mcf_clients}"
    
    # Check Client 0 updates
    if 0 in atk_clients:
        assert os.path.exists(os.path.join(attacked_dir, "client_updates", k, "client_00.pt"))
        assert not os.path.exists(os.path.join(mcf_dir, "client_updates", k, "client_00.pt"))

print("- Sampling exact. Client 0 properly excluded. No replacement.")

# Check NaNs/Infs and BatchNorm
m_cf_model = SmallCNN()
m_cf_model.load_state_dict(torch.load(os.path.join(mcf_dir, "checkpoints", "global_round_100.pt"), map_location="cpu"))
for name, p in m_cf_model.named_parameters():
    assert not torch.isnan(p).any(), f"NaN in {name}"
    assert not torch.isinf(p).any(), f"Inf in {name}"
print("- No NaNs/Infs in M_CF")

device = torch.device("cpu")
with open("results/stage7c_d2/unlearning/federaser/eval_indices.json", "r") as f:
    eval_indices = json.load(f)
    
_, test_loader = get_cifar10_dataloaders("./data", 64, False)
test_dataset = test_loader.dataset
eval_clean_dataset = torch.utils.data.Subset(test_dataset, eval_indices)
eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)

eval_clean_loader = torch.utils.data.DataLoader(eval_clean_dataset, batch_size=64, shuffle=False)
eval_asr_loader = torch.utils.data.DataLoader(eval_asr_dataset, batch_size=64, shuffle=False)

def evaluate(model_path):
    m = SmallCNN().to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    
    correct = 0
    total = 0
    asr_correct = 0
    asr_total = 0
    loss_sum = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in eval_clean_loader:
            x, y = x.to(device), y.to(device)
            out = m(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
                
        for x, y in eval_asr_loader:
            x, y = x.to(device), y.to(device)
            out = m(x)
            pred = out.argmax(dim=1)
            asr_correct += (pred == y).sum().item()
            asr_total += y.size(0)
            
    res = {
        "Clean Accuracy": 100.0 * correct / total,
        "Clean Loss": loss_sum / total,
        "ASR": 100.0 * asr_correct / asr_total if asr_total > 0 else 0
    }
    return res

print("Evaluating M_CF...")
mcf_res = evaluate(os.path.join(mcf_dir, "checkpoints", "global_round_100.pt"))
print(json.dumps(mcf_res, indent=4))

print("Evaluating Attacked...")
atk_res = evaluate(os.path.join(attacked_dir, "checkpoints", "global_round_100.pt"))
print(json.dumps(atk_res, indent=4))

