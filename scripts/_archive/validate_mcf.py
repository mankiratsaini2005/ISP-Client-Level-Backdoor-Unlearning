import json
import os
import torch
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset

mcf_dir = "results/stage7c_d2_mcf20260911_213425"
attack_dir = "results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423"

# 1. 100 rounds completed
assert os.path.exists(os.path.join(mcf_dir, "checkpoints/global_round_100.pt")), "global_round_100.pt missing!"

# 2. Client 0 never trained
with open(os.path.join(mcf_dir, "client_sampling.json")) as f:
    mcf_sampling = json.load(f)

with open(os.path.join(attack_dir, "client_sampling.json")) as f:
    attack_sampling = json.load(f)

for round_str, clients in mcf_sampling.items():
    assert 0 not in clients, f"Client 0 trained in {round_str}!"
    orig_clients = attack_sampling[round_str]
    expected_clients = [c for c in orig_clients if c != 0]
    assert clients == expected_clients, f"Sampling mismatch in {round_str}. MCF: {clients}, Attack expected: {expected_clients}"

# 3. Exact D2 partition used
with open(os.path.join(mcf_dir, "client_partitions.json")) as f:
    mcf_partition = json.load(f)
with open("results/stage7c_partitions/d2_distribution.json") as f:
    d2_partition = json.load(f)

for cid in d2_partition:
    mcf_key = f"client_{cid}" if f"client_{cid}" in mcf_partition else str(cid)
    assert mcf_partition[mcf_key]['num_samples'] == d2_partition[cid]['num_samples'], f"Partition mismatch for client {cid}!"
    
# 4. Evaluate M_CF_D2
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

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total, loss_sum / total

model = SmallCNN().to(device)
model.load_state_dict(torch.load(os.path.join(mcf_dir, "checkpoints/global_round_100.pt"), map_location=device))

clean_acc, clean_loss = evaluate(model, eval_clean_loader)
asr, _ = evaluate(model, eval_asr_loader)

res = {
    "Clean Accuracy": clean_acc,
    "Clean Loss": clean_loss,
    "ASR": asr
}
with open("results/stage7c_d2/mcf_validation.json", "w") as f:
    json.dump(res, f, indent=4)
print(json.dumps(res, indent=4))
print("M_CF Structural Checks PASS!")
