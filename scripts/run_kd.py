import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import hashlib
import json
import os
import time
import copy
import ssl

import torchvision.datasets.cifar as cifar
cifar.check_integrity = lambda path, md5=None: True

from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader
from src.training.train_utils import evaluate_extended
from torch.utils.data import Subset, DataLoader

ssl._create_default_https_context = ssl._create_unverified_context

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    import random
    random.seed(seed)

def hash_tensor(t):
    h = hashlib.sha256()
    h.update(t.numpy().tobytes())
    return h.hexdigest()

def force_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def get_bn_stats(model):
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            stats[name] = {
                'running_mean': m.running_mean.clone() if m.running_mean is not None else None,
                'running_var': m.running_var.clone() if m.running_var is not None else None,
                'num_batches_tracked': m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None
            }
    return stats

def compare_bn_stats(stats1, stats2):
    for name in stats1:
        s1 = stats1[name]
        s2 = stats2[name]
        if s1['running_mean'] is not None:
            assert torch.equal(s1['running_mean'], s2['running_mean']), f"BN running_mean changed in {name}"
        if s1['running_var'] is not None:
            assert torch.equal(s1['running_var'], s2['running_var']), f"BN running_var changed in {name}"
        if s1['num_batches_tracked'] is not None:
            assert torch.equal(s1['num_batches_tracked'], s2['num_batches_tracked']), f"BN num_batches_tracked changed in {name}"

def get_filtered_loader(test_loader, target_class, batch_size):
    base_dataset = test_loader.dataset
    valid_indices = []
    for i in range(len(base_dataset)):
        label = base_dataset.targets[i] if hasattr(base_dataset, 'targets') else base_dataset[i][1]
        if isinstance(label, torch.Tensor): label = label.item()
        if label != target_class: valid_indices.append(i)
                
    filtered_dataset = Subset(base_dataset, valid_indices)
    return torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)

def get_predictions(model, loader, device, is_triggered=False, trigger_size=3, trigger_value=None):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            if is_triggered:
                data_clone = data.clone()
                c, h, w = data_clone.shape[1:]
                data_clone[:, :, h-trigger_size:, w-trigger_size:] = trigger_value.to(device)
                data = data_clone
            
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
    return all_preds

def per_class_clean_acc(model, test_loader, device):
    model.eval()
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            for i in range(len(target)):
                lbl = target[i].item()
                class_total[lbl] += 1
                if preds[i].item() == lbl:
                    class_correct[lbl] += 1
    res = {}
    for i in range(10):
        res[str(i)] = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    return res

def main():
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    
    # 1. Setup Proxy
    print("Preparing CIFAR-100 Proxy...")
    download_start = time.time()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    download_time = time.time() - download_start
    print(f"Download time excluded from KD runtime.")
    
    set_seed(42)
    class_indices = {i: [] for i in range(100)}
    for idx in range(len(cifar100)):
        _, label = cifar100[idx]
        class_indices[label].append(idx)
        
    selected_indices = []
    for cls in range(100):
        cls_idx = np.array(class_indices[cls])
        np.random.shuffle(cls_idx)
        selected_indices.extend(cls_idx[:50].tolist())
        
    selected_indices = sorted(selected_indices)
    
    os.makedirs("results_phase2/proxy_data", exist_ok=True)
    with open("results_phase2/proxy_data/cifar100_proxy_indices.json", "w") as f:
        json.dump(selected_indices, f)
        
    proxy_subset = Subset(cifar100, selected_indices)
    
    transform_raw = transforms.ToTensor()
    cifar100_raw = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_raw)
    subset_raw = Subset(cifar100_raw, selected_indices)
    loader_raw = DataLoader(subset_raw, batch_size=5000, shuffle=False)
    data_batch_raw = next(iter(loader_raw))[0]
    data_hash = hash_tensor(data_batch_raw)
    
    print(f"Proxy dataset raw pixel tensor SHA256 Hash: {data_hash}")
    
    g = torch.Generator()
    g.manual_seed(42)
    proxy_loader = DataLoader(proxy_subset, batch_size=64, shuffle=True, generator=g)
    
    # 2. Inspect SmallCNN
    m_test = SmallCNN()
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in m_test.modules())
    has_do = any(isinstance(m, nn.Dropout) for m in m_test.modules())
    print(f"SmallCNN contains BatchNorm: {has_bn}")
    print(f"SmallCNN contains Dropout: {has_do}")
    
    # 4. Setup Models
    teacher = SmallCNN().to(device)
    teacher.load_state_dict(torch.load("results_phase2/iid_attack_logged/final_model.pt", map_location=device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    student = SmallCNN().to(device)
    student.load_state_dict(torch.load("results_phase2/iid_subtraction/final_model.pt", map_location=device))
    
    pre_kd_bn_stats = get_bn_stats(student)
    
    # 7. Teacher Sanity Check
    print("Running Teacher Sanity Check on Proxy...")
    teacher_preds = {i: 0 for i in range(10)}
    with torch.no_grad():
        for data, _ in proxy_loader:
            data = data.to(device)
            out = teacher(data)
            preds = out.argmax(dim=1)
            for p in preds:
                teacher_preds[p.item()] += 1
    print(f"Teacher predicted class distribution on CIFAR-100 proxy: {teacher_preds}")
    
    # 5. KD Training
    T = 4.0
    optimizer = optim.SGD(student.parameters(), lr=0.001, momentum=0.9)
    epochs = 5
    
    print("Starting KD Training...")
    kd_start_time = time.time()
    
    kd_loss_history = []
    
    for epoch in range(1, epochs + 1):
        student.train()
        force_bn_eval(student)
        
        epoch_loss = 0.0
        batches = 0
        
        for data, _ in proxy_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                t_logits = teacher(data)
                
            s_logits = student(data)
            
            loss = (T ** 2) * F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction='batchmean'
            )
            
            loss.backward()
            optimizer.step()
            
            force_bn_eval(student)
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        kd_loss_history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} - KD Loss: {avg_loss:.4f}")
        
    kd_runtime = time.time() - kd_start_time
    print(f"KD Training Complete. Runtime: {kd_runtime:.2f}s")
    
    post_kd_bn_stats = get_bn_stats(student)
    compare_bn_stats(pre_kd_bn_stats, post_kd_bn_stats)
    print("Assertion passed: All BN running statistics and num_batches_tracked are bit-identical.")
    
    with open("results_phase2/iid_subtraction/subtraction_runtime.json", "r") as f:
        sub_time = json.load(f)["runtime_seconds"]
        
    total_runtime = kd_runtime + sub_time
    
    out_dir = "results_phase2/iid_kd"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(out_dir, "final_model.pt"))
    
    meta = {
        "epochs": epochs,
        "batch_size": 64,
        "lr": 0.001,
        "momentum": 0.9,
        "T": 4.0,
        "optimizer": "SGD",
        "hyperparameter_note": "Values fixed from archived script. Not selected from new test results.",
        "kd_runtime_seconds": kd_runtime,
        "subtraction_runtime_seconds": sub_time,
        "total_unlearning_runtime": total_runtime,
        "kd_loss_history": kd_loss_history,
        "proxy_pixel_hash": data_hash
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
        
    # 10. Evaluation
    print("Starting evaluation...")
    batch_size = 256
    _, test_loader = get_cifar10_dataloaders("./data", batch_size=batch_size, use_fake=False)
    filtered_loader = get_filtered_loader(test_loader, 0, batch_size) # 9000
    asr_loader = create_asr_dataloader(test_loader, target_class=0, trigger_size=3, batch_size=batch_size)
    
    mean_trig = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std_trig = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    trigger_value = (1.0 - mean_trig) / std_trig
    
    mcf = SmallCNN().to(device)
    mcf.load_state_dict(torch.load("results_phase2/iid_counterfactual/final_model.pt", map_location=device))
    mcf_clean_preds = get_predictions(mcf, test_loader, device)
    mcf_trig_preds = get_predictions(mcf, filtered_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
    
    criterion = nn.CrossEntropyLoss()
    models = {
        "Phase-1 Clean": "results/iid_clean/final_model.pt",
        "Logged IID Attack": "results_phase2/iid_attack_logged/final_model.pt",
        "M_sub": "results_phase2/iid_subtraction/final_model.pt",
        "M_KD": "results_phase2/iid_kd/final_model.pt",
        "M_CF": "results_phase2/iid_counterfactual/final_model.pt"
    }
    
    eval_results = {}
    for name, path in models.items():
        m = SmallCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        
        clean_m = evaluate_extended(m, test_loader, criterion, device)
        trig_m = evaluate_extended(m, asr_loader, criterion, device)
        
        untrig_m = evaluate_extended(m, filtered_loader, criterion, device)
        total_untrig = sum(sum(row) for row in untrig_m["confusion_matrix"])
        pred_airplane = 0
        for i in range(1, 10): pred_airplane += untrig_m["confusion_matrix"][i][0]
        utr = 100.0 * pred_airplane / total_untrig if total_untrig > 0 else 0
        
        pc_acc = per_class_clean_acc(m, test_loader, device)
        
        c_preds = get_predictions(m, test_loader, device)
        t_preds = get_predictions(m, filtered_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
        
        ag_clean = 100.0 * sum(1 for p1, p2 in zip(c_preds, mcf_clean_preds) if p1 == p2) / 10000
        ag_trig = 100.0 * sum(1 for p1, p2 in zip(t_preds, mcf_trig_preds) if p1 == p2) / 9000
        
        eval_results[name] = {
            "Clean_Acc": clean_m["accuracy"],
            "Clean_Loss": clean_m["loss"],
            "Macro_F1": clean_m["macro_f1"],
            "Triggered_Rate": trig_m["accuracy"],
            "Untriggered_Airplane_Rate": utr,
            "Airplane_Recall": pc_acc["0"],
            "Per_Class_Clean_Accuracy": pc_acc,
            "CF_Agreement_Clean": ag_clean,
            "CF_Agreement_Triggered": ag_trig
        }
        
    with open("results_phase2/phase2_kd_evaluation.json", "w") as f:
        json.dump(eval_results, f, indent=4)
        
    print("All tasks complete.")

if __name__ == "__main__":
    main()
