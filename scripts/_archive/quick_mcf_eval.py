import torch
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
import json

device = torch.device("cpu")
_, test_loader = get_cifar10_dataloaders("./data", 64, False)

def eval_mcf(path, eval_idx_path):
    with open(eval_idx_path, 'r') as f:
        eval_indices = json.load(f)
    dataset = torch.utils.data.Subset(test_loader.dataset, eval_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    m = SmallCNN().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            out = m(x.to(device))
            correct += (out.argmax(1) == y.to(device)).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

c1_acc = eval_mcf("results/stage7b_c1_mcf_20260911_134941/checkpoints/global_round_100.pt", "results/stage7b_unlearning/c1/federaser/eval_indices.json")
c3_acc = eval_mcf("results/stage7b_c3_mcf_20260911_144140/checkpoints/global_round_100.pt", "results/stage7b_unlearning/c3/federaser/eval_indices.json")
print("C1 M_CF:", c1_acc)
print("C3 M_CF:", c3_acc)
