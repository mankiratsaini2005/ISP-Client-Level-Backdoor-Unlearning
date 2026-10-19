import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import PoisonedDataset

def create_filtered_asr_dataset(base_dataset, indices, target_class=0, trigger_size=3, trigger_loc=(29, 29)):
    """
    Creates an ASR evaluation dataset exactly matching previous stages, 
    but restricted ONLY to the provided indices.
    """
    subset = Subset(base_dataset, indices)
    
    # Filter out samples already belonging to the target class
    valid_indices = []
    for i in range(len(subset)):
        _, target = subset[i]
        if target != target_class:
            valid_indices.append(i)
            
    filtered_subset = Subset(subset, valid_indices)
    poisoned_dataset = PoisonedDataset(filtered_subset, 
                                       poison_indices=list(range(len(filtered_subset))),
                                       target_class=target_class, 
                                       trigger_size=trigger_size)
    return poisoned_dataset

def evaluate_subset(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    avg_loss = total_loss / total
    return avg_loss, acc

def run_kd(teacher_path, student_path, reference_path, out_dir):
    start_time = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    
    # --- 1. Dataset Construction ---
    print("Preparing datasets...")
    _, test_loader_full = get_cifar10_dataloaders(batch_size=64)
    test_dataset = test_loader_full.dataset
    
    # Deterministic split
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(test_dataset), generator=g).tolist()
    
    proxy_indices = indices[:2500]
    eval_indices = indices[2500:]
    
    # Save indices
    with open(os.path.join(out_dir, "proxy_indices.json"), "w") as f:
        json.dump(proxy_indices, f)
    with open(os.path.join(out_dir, "eval_indices.json"), "w") as f:
        json.dump(eval_indices, f)
        
    proxy_dataset = Subset(test_dataset, proxy_indices)
    eval_clean_dataset = Subset(test_dataset, eval_indices)
    eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices)
    
    proxy_loader = DataLoader(proxy_dataset, batch_size=64, shuffle=True)
    eval_clean_loader = DataLoader(eval_clean_dataset, batch_size=64, shuffle=False)
    eval_asr_loader = DataLoader(eval_asr_dataset, batch_size=64, shuffle=False)
    
    print(f"Proxy dataset size: {len(proxy_dataset)}")
    print(f"Eval clean size: {len(eval_clean_dataset)}")
    print(f"Eval ASR size: {len(eval_asr_dataset)}")
    
    # --- 2. Model Loading ---
    print("Loading models...")
    teacher = SmallCNN().to(device)
    teacher.load_state_dict(torch.load(teacher_path, weights_only=False))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    student = SmallCNN().to(device)
    student.load_state_dict(torch.load(student_path, weights_only=False))
    
    # --- 3. Optimization Setup ---
    T = 4.0
    optimizer = torch.optim.SGD(student.parameters(), lr=0.001, momentum=0.9)
    
    # --- 4. Smoke Test ---
    print("Running 1-batch smoke test...")
    student.train()
    smoke_inputs, smoke_labels = next(iter(proxy_loader))
    smoke_inputs = smoke_inputs.to(device)
    
    teacher_pre = {name: param.clone() for name, param in teacher.named_parameters()}
    student_pre = {name: param.clone() for name, param in student.named_parameters()}
    
    with torch.no_grad():
        t_logits = teacher(smoke_inputs)
    s_logits = student(smoke_inputs)
    
    loss = (T**2) * F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction='batchmean'
    )
    
    if not torch.isfinite(loss):
        raise ValueError("Smoke Test Failed: KD Loss is NaN/Inf")
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Verification
    for name, param in teacher.named_parameters():
        if not torch.equal(param, teacher_pre[name]):
            raise ValueError(f"Smoke Test Failed: Teacher parameter {name} changed!")
            
    student_changed = False
    for name, param in student.named_parameters():
        if not torch.equal(param, student_pre[name]):
            student_changed = True
            break
    if not student_changed:
        raise ValueError("Smoke Test Failed: Student parameters did not update!")
        
    print("Smoke test passed successfully!")
    
    # Reset student to pure Stage 5A state for the real run
    student.load_state_dict(torch.load(student_path, weights_only=False))
    
    # --- 5. Evaluation & Training Loop ---
    metrics = []
    
    def eval_and_log(epoch, kd_loss=None, epoch_time=0):
        c_loss, c_acc = evaluate_subset(student, eval_clean_loader, device)
        _, asr = evaluate_subset(student, eval_asr_loader, device)
        
        m = {
            "epoch": epoch,
            "kd_loss": kd_loss,
            "clean_loss": c_loss,
            "clean_acc": c_acc,
            "asr": asr,
            "epoch_time_s": epoch_time
        }
        metrics.append(m)
        print(f"Epoch {epoch}: Clean Acc {c_acc:.2f}%, ASR {asr:.2f}%, KD Loss: {kd_loss if kd_loss else 'N/A'}")
        return m

    # Epoch 0 (Baseline Stage 5A)
    print("Evaluating Epoch 0 (Stage 5A Starting Point)...")
    eval_and_log(0)
    
    total_kd_time = 0.0
    
    for epoch in range(1, 6):
        student.train()
        epoch_start = time.time()
        running_loss = 0.0
        
        for inputs, _ in proxy_loader:  # Deliberately ignoring labels
            inputs = inputs.to(device)
            
            with torch.no_grad():
                t_logits = teacher(inputs)
            
            optimizer.zero_grad()
            s_logits = student(inputs)
            
            loss = (T**2) * F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction='batchmean'
            )
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_time = time.time() - epoch_start
        total_kd_time += epoch_time
        avg_loss = running_loss / len(proxy_dataset)
        
        eval_and_log(epoch, avg_loss, epoch_time)
        
    # --- 6. Reference Models Evaluation (Fair Comparison) ---
    print("Evaluating Compromised (Teacher) and Reference models on identical subset...")
    # Compromised Teacher
    ref_3B = SmallCNN().to(device)
    ref_3B.load_state_dict(torch.load(teacher_path, weights_only=False))
    l_3b, a_3b = evaluate_subset(ref_3B, eval_clean_loader, device)
    _, asr_3b = evaluate_subset(ref_3B, eval_asr_loader, device)
    
    # Counterfactual Reference
    ref_4 = SmallCNN().to(device)
    ref_4.load_state_dict(torch.load(reference_path, weights_only=False))
    l_4, a_4 = evaluate_subset(ref_4, eval_clean_loader, device)
    _, asr_4 = evaluate_subset(ref_4, eval_asr_loader, device)
    
    print(f"Teacher (Subset): Clean Acc {a_3b:.2f}%, ASR {asr_3b:.2f}%")
    print(f"Reference (Subset): Clean Acc {a_4:.2f}%, ASR {asr_4:.2f}%")
    
    # --- 7. Save Final Artifacts ---
    torch.save(student.state_dict(), os.path.join(out_dir, "checkpoints", "final_unlearned.pt"))
    
    # Save metadata & metrics
    metadata = {
        "teacher_path": teacher_path,
        "student_path": student_path,
        "reference_path": reference_path,
        "temperature": T,
        "lr": 0.001,
        "momentum": 0.9,
        "epochs": 5,
        "batch_size": 64,
        "proxy_size": len(proxy_dataset),
        "eval_size": len(eval_clean_dataset),
        "total_kd_time_s": total_kd_time,
        "additional_communication_bytes": 0,
        "reference_subset_metrics": {
            "Stage_3B": {"clean_acc": a_3b, "clean_loss": l_3b, "asr": asr_3b},
            "Stage_4": {"clean_acc": a_4, "clean_loss": l_4, "asr": asr_4}
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    import pandas as pd
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    
    # --- 8. Generate Plots ---
    plot_dir = os.path.join(out_dir, "figures")
    epochs = df["epoch"].tolist()
    clean_accs = df["clean_acc"].tolist()
    asrs = df["asr"].tolist()
    kd_losses = df["kd_loss"].tolist()
    
    # Plot 1: Clean Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, clean_accs, marker='o', label='Stage 5B (KD)')
    plt.axhline(y=a_4, color='g', linestyle='--', label='Stage 4 (Gold Reference)')
    plt.axhline(y=a_3b, color='r', linestyle=':', label='Stage 3B (Backdoored Baseline)')
    plt.title('Clean Accuracy Recovery vs KD Epochs')
    plt.xlabel('KD Epoch (0 = Stage 5A Subtraction)')
    plt.ylabel('Clean Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "accuracy_recovery.png"), dpi=300)
    plt.close()
    
    # Plot 2: ASR
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, asrs, marker='o', color='red', label='Stage 5B (KD)')
    plt.axhline(y=asr_4, color='g', linestyle='--', label='Stage 4 (Gold Reference)')
    plt.axhline(y=asr_3b, color='orange', linestyle=':', label='Stage 3B (Backdoored Baseline)')
    plt.title('Attack Success Rate (ASR) vs KD Epochs')
    plt.xlabel('KD Epoch (0 = Stage 5A Subtraction)')
    plt.ylabel('ASR (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "asr_recovery.png"), dpi=300)
    plt.close()
    
    # Plot 3: KD Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs[1:], kd_losses[1:], marker='o', color='purple')
    plt.title('KD Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "kd_loss.png"), dpi=300)
    plt.close()
    
    # Plot 4: Acc vs ASR trajectory
    plt.figure(figsize=(8, 6))
    plt.scatter(clean_accs, asrs, c=epochs, cmap='viridis', label='Stage 5B Trajectory')
    for i, txt in enumerate(epochs):
        plt.annotate(f'Ep {txt}', (clean_accs[i], asrs[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.scatter([a_3b], [asr_3b], color='red', marker='X', s=100, label='Stage 3B (Start)')
    plt.scatter([a_4], [asr_4], color='green', marker='*', s=150, label='Stage 4 (Ideal)')
    
    plt.title('Unlearning Trajectory: Clean Accuracy vs ASR')
    plt.xlabel('Clean Accuracy (%)')
    plt.ylabel('ASR (%)')
    plt.colorbar(label='KD Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "accuracy_vs_asr.png"), dpi=300)
    plt.close()
    
    print(f"\nStage 5B complete. All outputs saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    run_kd(args.teacher_path, args.student_path, args.reference_path, args.out_dir)
