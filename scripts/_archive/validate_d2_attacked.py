import os
import torch
import json
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
model_path = "results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423/checkpoints/global_round_100.pt"
model.load_state_dict(torch.load(model_path, map_location=device))

clean_acc, clean_loss = evaluate(model, eval_clean_loader)
asr, _ = evaluate(model, eval_asr_loader)

print(json.dumps({
    "Clean Accuracy": clean_acc,
    "Clean Loss": clean_loss,
    "ASR": asr
}, indent=4))
