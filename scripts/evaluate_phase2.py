import torch
import torch.nn as nn
import json
import os
import copy
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader
from src.training.train_utils import evaluate_extended
from torch.utils.data import Subset

def create_untriggered_baseline_loader(test_loader, target_class, batch_size):
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

def per_class_asr(model, test_loader, device, trigger_size, trigger_value):
    model.eval()
    class_totals = {i: 0 for i in range(1, 10)}
    class_attacked = {i: 0 for i in range(1, 10)}
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            data_clone = data.clone()
            c, h, w = data_clone.shape[1:]
            data_clone[:, :, h-trigger_size:, w-trigger_size:] = trigger_value.to(device)
            
            output = model(data_clone)
            preds = output.argmax(dim=1)
            
            for i in range(len(target)):
                orig_label = target[i].item()
                if orig_label != 0:
                    class_totals[orig_label] += 1
                    if preds[i].item() == 0:
                        class_attacked[orig_label] += 1
                        
    res = {}
    for i in range(1, 10):
        res[str(i)] = 100.0 * class_attacked[i] / class_totals[i] if class_totals[i] > 0 else 0
    return res

def main():
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    with open("configs/v2_iid.yaml", "r") as f:
        import yaml
        config = yaml.safe_load(f)
        
    batch_size = 256
    _, test_loader = get_cifar10_dataloaders("./data", batch_size=batch_size, use_fake=False)
    asr_loader = create_asr_dataloader(test_loader, target_class=0, trigger_size=3, batch_size=batch_size)
    untriggered_loader = create_untriggered_baseline_loader(test_loader, target_class=0, batch_size=batch_size)
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    trigger_value = (1.0 - mean) / std
    
    models = {
        "Phase-1 Clean": "results/iid_clean/final_model.pt",
        "Logged IID Attack": "results_phase2/iid_attack_logged/final_model.pt",
        "M_sub": "results_phase2/iid_subtraction/final_model.pt",
        "M_CF": "results_phase2/iid_counterfactual/final_model.pt"
    }
    
    loaded_models = {}
    for name, path in models.items():
        m = SmallCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        loaded_models[name] = m
        
    # Get M_CF predictions for agreement
    mcf_clean_preds = get_predictions(loaded_models["M_CF"], test_loader, device)
    mcf_triggered_preds = get_predictions(loaded_models["M_CF"], test_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for name, m in loaded_models.items():
        print(f"Evaluating {name}...")
        clean_metrics = evaluate_extended(m, test_loader, criterion, device)
        asr_metrics = evaluate_extended(m, asr_loader, criterion, device)
        untrig_metrics = evaluate_extended(m, untriggered_loader, criterion, device)
        
        # Untriggered airplane rate
        total_untrig = sum(sum(row) for row in untrig_metrics["confusion_matrix"])
        pred_airplane = 0
        for i in range(1, 10):
            pred_airplane += untrig_metrics["confusion_matrix"][i][0]
        actual_total = total_untrig - sum(untrig_metrics["confusion_matrix"][0]) # non-airplanes
        utr = 100.0 * pred_airplane / actual_total if actual_total > 0 else 0
        
        # Per class ASR
        per_class = per_class_asr(m, test_loader, device, 3, trigger_value)
        
        # Agreement
        clean_preds = get_predictions(m, test_loader, device)
        trig_preds = get_predictions(m, test_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
        
        agree_clean = 100.0 * sum(1 for p1, p2 in zip(clean_preds, mcf_clean_preds) if p1 == p2) / 10000
        agree_trig = 100.0 * sum(1 for p1, p2 in zip(trig_preds, mcf_triggered_preds) if p1 == p2) / 10000
        
        results[name] = {
            "Clean_Acc": clean_metrics["accuracy"],
            "Clean_Loss": clean_metrics["loss"],
            "Macro_F1": clean_metrics["macro_f1"],
            "Triggered_Rate": asr_metrics["accuracy"],
            "Untriggered_Airplane_Rate": utr,
            "Per_Class_ASR": per_class,
            "CF_Agreement_Clean": agree_clean,
            "CF_Agreement_Triggered": agree_trig
        }
        
    with open("results_phase2/phase2_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
