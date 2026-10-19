import torch
import json
import torch.nn as nn
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.training.train_utils import evaluate_extended
from src.data.backdoor import create_asr_dataloader
from torch.utils.data import Subset

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
    batch_size = 256
    
    _, test_loader = get_cifar10_dataloaders("./data", batch_size=batch_size, use_fake=False)
    filtered_loader = get_filtered_loader(test_loader, 0, batch_size) # 9000 images
    asr_loader = create_asr_dataloader(test_loader, target_class=0, trigger_size=3, batch_size=batch_size)
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    trigger_value = (1.0 - mean) / std
    
    models = {
        "Phase-1 Non-IID Clean": "results/noniid_clean/final_model.pt",
        "Logged Non-IID Attack": "results_phase2/noniid_attack_logged/final_model.pt",
        "Non-IID M_sub": "results_phase2/noniid_subtraction/final_model.pt",
        "Non-IID M_CF": "results_phase2/noniid_counterfactual/final_model.pt"
    }
    
    loaded_models = {}
    for name, path in models.items():
        m = SmallCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        loaded_models[name] = m
        
    mcf_clean_preds = get_predictions(loaded_models["Non-IID M_CF"], test_loader, device)
    mcf_trig_preds = get_predictions(loaded_models["Non-IID M_CF"], filtered_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
    
    results = {}
    criterion = nn.CrossEntropyLoss()
    
    for name, m in loaded_models.items():
        print(f"Evaluating {name}...")
        
        clean_m = evaluate_extended(m, test_loader, criterion, device)
        trig_m = evaluate_extended(m, asr_loader, criterion, device)
        untrig_m = evaluate_extended(m, filtered_loader, criterion, device)
        
        total_untrig = sum(sum(row) for row in untrig_m["confusion_matrix"])
        pred_airplane = 0
        for i in range(1, 10): pred_airplane += untrig_m["confusion_matrix"][i][0]
        utr = 100.0 * pred_airplane / total_untrig if total_untrig > 0 else 0
        
        # Per class clean acc
        pc_acc = per_class_clean_acc(m, test_loader, device)
        
        # Agreement
        clean_preds = get_predictions(m, test_loader, device)
        trig_preds = get_predictions(m, filtered_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
        
        agree_clean = 100.0 * sum(1 for p1, p2 in zip(clean_preds, mcf_clean_preds) if p1 == p2) / 10000
        agree_trig = 100.0 * sum(1 for p1, p2 in zip(trig_preds, mcf_trig_preds) if p1 == p2) / 9000
        
        results[name] = {
            "Clean_Acc": clean_m["accuracy"],
            "Clean_Loss": clean_m["loss"],
            "Macro_F1": clean_m["macro_f1"],
            "Triggered_Rate": trig_m["accuracy"],
            "Untriggered_Airplane_Rate": utr,
            "Airplane_Recall": pc_acc["0"],
            "Per_Class_Clean_Accuracy": pc_acc,
            "CF_Agreement_Clean": agree_clean,
            "CF_Agreement_Triggered": agree_trig
        }
        
    with open("results_phase2/noniid_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Non-IID evaluation complete.")

if __name__ == "__main__":
    main()
