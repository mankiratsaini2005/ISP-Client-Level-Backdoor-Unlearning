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

def calculate_macro_f1(targets, preds, num_classes=10):
    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(targets, preds) if t == c and p == c)
        fp = sum(1 for t, p in zip(targets, preds) if t != c and p == c)
        fn = sum(1 for t, p in zip(targets, preds) if t == c and p != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    return (sum(f1_scores) / num_classes) * 100.0

def run_eval(setting_name, m_cf_path):
    out_dir = "results_phase2/federaser/corrected_evaluation"
    
    _, test_loader_full = get_cifar10_dataloaders(batch_size=batch_size, use_fake=False)
    test_dataset = test_loader_full.dataset
    
    eval_clean_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    non_airplane_indices = [i for i, label in enumerate(test_dataset.targets) if label != 0]
    eval_non_airplane_dataset = Subset(test_dataset, non_airplane_indices)
    
    eval_asr_loader = create_asr_dataloader(test_loader_full, batch_size=batch_size, target_class=0, trigger_size=3)
    eval_untriggered_loader = DataLoader(eval_non_airplane_dataset, batch_size=batch_size, shuffle=False)
    
    airplane_indices = [i for i, label in enumerate(test_dataset.targets) if label == 0]
    eval_airplane_dataset = Subset(test_dataset, airplane_indices)
    eval_airplane_loader = DataLoader(eval_airplane_dataset, batch_size=batch_size, shuffle=False)

    global_model = SmallCNN().to(device)
    global_model.load_state_dict(torch.load(f"results_phase2/federaser/{setting_name}/final_model.pt", map_location=device))
    global_model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    clean_loss, clean_acc = evaluate(global_model, eval_clean_loader, criterion, device)
    _, asr = evaluate(global_model, eval_asr_loader, criterion, device)
    
    utr_correct = 0
    utr_total = 0
    with torch.no_grad():
        for x, _ in eval_untriggered_loader:
            x = x.to(device)
            p = global_model(x).argmax(1)
            utr_correct += (p == 0).sum().item()
            utr_total += x.size(0)
    utr = (utr_correct / utr_total) * 100.0 if utr_total > 0 else 0.0
    
    _, airplane_recall = evaluate(global_model, eval_airplane_loader, criterion, device)
    
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
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in eval_clean_loader:
            x = x.to(device)
            p = global_model(x).argmax(1)
            all_preds.extend(p.cpu().tolist())
            all_targets.extend(y.tolist())
            
    macro_f1 = calculate_macro_f1(all_targets, all_preds)
    
    eval_results = {
        "Clean_Accuracy": clean_acc,
        "Clean_Loss": clean_loss,
        "Triggered_Rate": asr,
        "Untriggered_Airplane_Rate": utr,
        "Clean_CF_Agreement": cf_clean,
        "Triggered_CF_Agreement": cf_trig,
        "Airplane_Recall": airplane_recall,
        "Macro_F1": macro_f1
    }
    with open(os.path.join(out_dir, f"{setting_name}_corrected_evaluation.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"{setting_name} Evaluated!")

if __name__ == "__main__":
    run_eval("iid", "results_phase2/iid_counterfactual/final_model.pt")
    run_eval("noniid", "results_phase2/noniid_counterfactual/final_model.pt")
