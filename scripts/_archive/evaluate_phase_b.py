import torch
import os
import glob
import pandas as pd
import json
import time

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
sys.path.append(os.getcwd())
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Find the latest Phase B run
dirs = glob.glob("results/stage6d_phase_b_full_*")
latest_dir = max(dirs, key=os.path.getmtime)
print(f"Evaluating {latest_dir}")

base_dir = "results/stage_6d_temporal"
ref_path = os.path.join(base_dir, "matched_counterfactual/checkpoints/global_round_100.pt")
no_attack_path = os.path.join(base_dir, "no_attack/checkpoints/global_round_100.pt")

# Load evaluation data
_, test_loader_full = get_cifar10_dataloaders(batch_size=64)
test_dataset = test_loader_full.dataset

g = torch.Generator().manual_seed(42)
indices = torch.randperm(len(test_dataset), generator=g).tolist()
eval_indices = indices[2500:]

eval_clean_dataset = Subset(test_dataset, eval_indices)
eval_clean_loader = DataLoader(eval_clean_dataset, batch_size=256, shuffle=False)
eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)
eval_asr_loader = DataLoader(eval_asr_dataset, batch_size=256, shuffle=False)

def get_model(path):
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def evaluate_subset(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def get_predictions(model, dataloader):
    preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.append(outputs.argmax(dim=1))
    return torch.cat(preds)

def calc_agreement(p1, p2):
    return (p1 == p2).float().mean().item() * 100

ref_model = get_model(ref_path)
preds_ref = get_predictions(ref_model, eval_clean_loader)

# Load teacher models to get baseline ASR/Acc
teachers = {
    "early": get_model(os.path.join(base_dir, "early/attack/checkpoints/global_round_100.pt")),
    "middle": get_model(os.path.join(base_dir, "middle/attack/checkpoints/global_round_100.pt")),
    "late": get_model(os.path.join(base_dir, "late/attack/checkpoints/global_round_100.pt")),
}

methods = {
    "subtraction": "checkpoints/subtraction_model.pt",
    "kd": "checkpoints/final_unlearned.pt",
    "federaser": "checkpoints/global_round_100.pt"
}

results = []
for cond in ["early", "middle", "late"]:
    t_acc = evaluate_subset(teachers[cond], eval_clean_loader)
    t_asr = evaluate_subset(teachers[cond], eval_asr_loader)
    
    for method_name, checkpoint_name in methods.items():
        model_path = os.path.join(latest_dir, cond, method_name, checkpoint_name)
        if not os.path.exists(model_path):
            print(f"Missing {model_path}")
            continue
            
        model = get_model(model_path)
        c_acc = evaluate_subset(model, eval_clean_loader)
        c_asr = evaluate_subset(model, eval_asr_loader)
        
        preds_model = get_predictions(model, eval_clean_loader)
        fidelity = calc_agreement(preds_model, preds_ref)
        
        # Load time/comm from metadata
        runtime = 0.0
        meta_path = os.path.join(latest_dir, cond, method_name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                if method_name == "subtraction":
                    runtime = meta.get("total_unlearning_time_s", 0)
                elif method_name == "kd":
                    runtime = meta.get("total_kd_time_s", 0)
                elif method_name == "federaser":
                    runtime = meta.get("total_time_s", 0)
        
        results.append({
            "Condition": cond.capitalize(),
            "Method": method_name.upper(),
            "Pre-Unlearn Acc (%)": f"{t_acc:.2f}",
            "Post-Unlearn Acc (%)": f"{c_acc:.2f}",
            "Pre-Unlearn ASR (%)": f"{t_asr:.2f}",
            "Post-Unlearn ASR (%)": f"{c_asr:.2f}",
            "Fidelity to MCF (%)": f"{fidelity:.2f}",
            "Runtime (s)": f"{runtime:.1f}"
        })

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))
df.to_csv(os.path.join(latest_dir, "consolidated_metrics.csv"), index=False)
