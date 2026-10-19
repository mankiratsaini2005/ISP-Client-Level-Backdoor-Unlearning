import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.data import DataLoader, Subset

from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset
from src.data.backdoor import PoisonedDataset

def compute_l2_distance(state1, state2):
    dist_sq = 0.0
    for k in state1.keys():
        if state1[k].is_floating_point():
            dist_sq += torch.sum((state1[k] - state2[k]) ** 2).item()
    return np.sqrt(dist_sq)

def compute_norm(state):
    norm_sq = 0.0
    for k in state.keys():
        if state[k].is_floating_point():
            norm_sq += torch.sum(state[k] ** 2).item()
    return np.sqrt(norm_sq)

def compute_layerwise_distance(state_u, state_r):
    layers = []
    distances = []
    for k in state_r.keys():
        if state_r[k].is_floating_point() and 'weight' in k:
            dist = torch.norm(state_u[k] - state_r[k], p=2).item()
            norm_r = torch.norm(state_r[k], p=2).item()
            layers.append(k)
            distances.append(dist / (norm_r + 1e-8))
    return layers, distances

def jensen_shannon_div(p, q):
    p = p + 1e-8
    q = q + 1e-8
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * F.kl_div(p.log(), m, reduction='batchmean') + 0.5 * F.kl_div(q.log(), m, reduction='batchmean')

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths (adjust to match actual)
    dir_M_B = "results/attack_scaling_final_gamma5_20260909_015903"
    dir_M_R = "results/full_retraining_final_20260909_024627"
    dir_M_S = "results/unlearning_5a_subtraction_20260909_131440"
    dir_M_KD = "results/unlearning_5b_kd_20260909_183734"
    dir_M_FE = "results/unlearning_federaser_20260909_224318"
    
    models = {
        'M_B': os.path.join(dir_M_B, "checkpoints", "global_round_100.pt"),
        'M_R': os.path.join(dir_M_R, "checkpoints", "global_round_100.pt"),
        'M_S': os.path.join(dir_M_S, "checkpoints", "subtraction_model.pt"),
        'M_KD': os.path.join(dir_M_KD, "checkpoints", "final_unlearned.pt"),
        'M_FE': os.path.join(dir_M_FE, "checkpoints", "global_round_100.pt")
    }
    
    states = {}
    for name, path in models.items():
        if os.path.exists(path):
            states[name] = torch.load(path, map_location=device, weights_only=True)
            print(f"Loaded {name}")
        else:
            print(f"Warning: {path} not found.")

    # A. Parameter Space Analysis
    print("\n--- PARAMETER SPACE ANALYSIS ---")
    norm_R = compute_norm(states['M_R'])
    print(f"Norm of M_R: {norm_R:.4f}")
    
    layerwise_dists = {}
    for m_name in ['M_S', 'M_KD', 'M_FE']:
        if m_name not in states: continue
        d_R = compute_l2_distance(states[m_name], states['M_R'])
        d_B = compute_l2_distance(states[m_name], states['M_B'])
        rel_d_R = d_R / norm_R
        print(f"{m_name}: ||U-R||2 = {d_R:.4f} (rel {rel_d_R:.4f}), ||U-B||2 = {d_B:.4f}")
        
        layers, dists = compute_layerwise_distance(states[m_name], states['M_R'])
        layerwise_dists[m_name] = dists
    
    # Plot layerwise distances
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(layers))
    width = 0.25
    if 'M_S' in layerwise_dists:
        plt.bar(x - width, layerwise_dists['M_S'], width, label='M_S (Naive)')
    if 'M_KD' in layerwise_dists:
        plt.bar(x, layerwise_dists['M_KD'], width, label='M_KD (Repair)')
    if 'M_FE' in layerwise_dists:
        plt.bar(x + width, layerwise_dists['M_FE'], width, label='M_FE (FedEraser)')
    plt.xticks(x, layers, rotation=45, ha="right")
    plt.ylabel("Relative L2 Distance to M_R")
    plt.title("Layer-wise Parameter Distance to Counterfactual Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/layerwise_counterfactual_distance.png")
    print("Saved figures/layerwise_counterfactual_distance.png")
    
    # B. Dataloaders
    _, test_loader = get_cifar10_dataloaders(batch_size=200)
    test_dataset = test_loader.dataset
    
    with open(os.path.join(dir_M_KD, "eval_indices.json"), "r") as f:
        eval_indices = json.load(f)
        
    eval_subset = Subset(test_dataset, eval_indices)
    eval_clean_loader = DataLoader(eval_subset, batch_size=200, shuffle=False)
    
    # ASR loader (original)
    asr_dataset_orig = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)
    asr_loader_orig = DataLoader(asr_dataset_orig, batch_size=200, shuffle=False)
    
    # C. Prediction Level Fidelity
    print("\n--- PREDICTION LEVEL FIDELITY & PER CLASS ---")
    
    def get_predictions(model_state):
        model = SmallCNN().to(device)
        model.load_state_dict(model_state)
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for X, y in eval_clean_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())
        return torch.cat(all_preds), torch.cat(all_probs), torch.cat(all_targets)
        
    preds_R, probs_R, targets = get_predictions(states['M_R'])
    
    for m_name in ['M_B', 'M_S', 'M_KD', 'M_FE', 'M_R']:
        if m_name not in states: continue
        preds_U, probs_U, _ = get_predictions(states[m_name])
        
        acc = (preds_U == targets).float().mean().item() * 100
        agreement = (preds_U == preds_R).float().mean().item() * 100
        
        js_div = jensen_shannon_div(probs_U, probs_R).item()
        
        print(f"{m_name:4s}: Acc = {acc:.2f}%, Agreement with M_R = {agreement:.2f}%, JS Div = {js_div:.4f}")
        
    # D. Trigger Variants Analysis
    print("\n--- TRIGGER VARIANTS ANALYSIS (ASR) ---")
    # 1. Spatial Shift (top-left instead of bottom-right)
    class ShiftedTriggerDataset(PoisonedDataset):
        def _apply_trigger(self, image):
            img = image.clone()
            img[:, 0:3, 0:3] = 2.5
            return img
            
    filtered_targets = [i for i in eval_indices if test_dataset[i][1] != 0]
    sub = Subset(test_dataset, filtered_targets)
    asr_dataset_shifted = ShiftedTriggerDataset(sub, poison_indices=list(range(len(sub))), target_class=0, trigger_size=3)
    asr_loader_shifted = DataLoader(asr_dataset_shifted, batch_size=200, shuffle=False)
    
    # 2. Patch Size Variation (4x4)
    asr_dataset_large = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=4)
    asr_loader_large = DataLoader(asr_dataset_large, batch_size=200, shuffle=False)
    
    def get_asr(model_state, loader):
        model = SmallCNN().to(device)
        model.load_state_dict(model_state)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return (correct / total) * 100 if total > 0 else 0
        
    print(f"{'Model':<6} | {'Orig (3x3 BR)':<15} | {'Shift (3x3 TL)':<15} | {'Size (4x4 BR)':<15}")
    print("-" * 55)
    for m_name in ['M_B', 'M_S', 'M_KD', 'M_FE', 'M_R']:
        if m_name not in states: continue
        asr_orig = get_asr(states[m_name], asr_loader_orig)
        asr_shift = get_asr(states[m_name], asr_loader_shifted)
        asr_large = get_asr(states[m_name], asr_loader_large)
        print(f"{m_name:<6} | {asr_orig:<15.2f} | {asr_shift:<15.2f} | {asr_large:<15.2f}")

if __name__ == "__main__":
    main()
