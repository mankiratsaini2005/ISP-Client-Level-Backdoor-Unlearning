import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.training.train_utils import evaluate
from src.data.backdoor import create_asr_dataloader

device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

batch_size = 64

def run_eval(setting_name, m_cf_path):
    out_dir = f"results_phase2/federaser/{setting_name}"
    
    train_loader_full, test_loader_full = get_cifar10_dataloaders(batch_size=batch_size, use_fake=False)
    
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(test_loader_full.dataset), generator=g).tolist()
    eval_indices = indices[2500:]
    eval_clean_dataset = Subset(test_loader_full.dataset, eval_indices)
    eval_clean_loader = DataLoader(eval_clean_dataset, batch_size=batch_size, shuffle=False)
    
    eval_asr_loader = create_asr_dataloader(test_loader_full, batch_size=batch_size, target_class=0, trigger_size=3)
    
    # UTR: Non-airplanes predicted as Airplanes
    non_airplane_indices = [idx for idx in eval_indices if test_loader_full.dataset.targets[idx] != 0]
    eval_non_airplane_dataset = Subset(test_loader_full.dataset, non_airplane_indices)
    eval_non_airplane_loader = DataLoader(eval_non_airplane_dataset, batch_size=batch_size, shuffle=False)

    global_model = SmallCNN().to(device)
    global_model.load_state_dict(torch.load(os.path.join(out_dir, "final_model.pt"), map_location=device))
    
    criterion = nn.CrossEntropyLoss()
    
    clean_loss, clean_acc = evaluate(global_model, eval_clean_loader, criterion, device)
    _, asr = evaluate(global_model, eval_asr_loader, criterion, device)
    
    # Evaluate UTR (predicting class 0 on non-class 0 images)
    utr_correct = 0
    utr_total = 0
    global_model.eval()
    with torch.no_grad():
        for x, _ in eval_non_airplane_loader:
            x = x.to(device)
            p = global_model(x).argmax(1)
            utr_correct += (p == 0).sum().item()
            utr_total += x.size(0)
    utr = (utr_correct / utr_total) * 100.0 if utr_total > 0 else 0.0
    
    m_cf = SmallCNN().to(device)
    m_cf.load_state_dict(torch.load(m_cf_path, map_location=device))
    m_cf.eval()
    
    def get_cf_agreement(loader):
        agree, total = 0, 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                p_fed = global_model(x).argmax(1)
                p_cf = m_cf(x).argmax(1)
                agree += (p_fed == p_cf).sum().item()
                total += x.size(0)
        return (agree / total) * 100.0 if total > 0 else 0.0

    cf_clean = get_cf_agreement(eval_clean_loader)
    cf_trig = get_cf_agreement(eval_asr_loader)
    
    correct = {i: 0 for i in range(10)}
    total_c = {i: 0 for i in range(10)}
    with torch.no_grad():
        for x, y in test_loader_full:
            x, y = x.to(device), y.to(device)
            p = global_model(x).argmax(1)
            for pred, true_label in zip(p, y):
                if pred.item() == true_label.item():
                    correct[true_label.item()] += 1
                total_c[true_label.item()] += 1
    
    per_class_acc = {i: (correct[i]/total_c[i])*100.0 for i in range(10)}
    macro_f1 = sum(per_class_acc.values()) / 10.0
    
    eval_results = {
        "Clean_Accuracy": clean_acc,
        "Clean_Loss": clean_loss,
        "Triggered_Rate": asr,
        "Untriggered_Airplane_Rate": utr,
        "Clean_CF_Agreement": cf_clean,
        "Triggered_CF_Agreement": cf_trig,
        "Airplane_Recall": per_class_acc[0],
        "Macro_F1": macro_f1,
        "Per_Class_Clean_Accuracy": per_class_acc
    }
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"{setting_name} Evaluated!")

if __name__ == "__main__":
    run_eval("iid", "results_phase2/iid_counterfactual/final_model.pt")
    run_eval("noniid", "results_phase2/noniid_counterfactual/final_model.pt")
