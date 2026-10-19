import torch
import json
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
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

def get_predictions_and_ttr(model, loader, device, is_triggered=False, trigger_size=3, trigger_value=None):
    model.eval()
    all_preds = []
    ttr_count = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            if is_triggered:
                data_clone = data.clone()
                c, h, w = data_clone.shape[1:]
                data_clone[:, :, h-trigger_size:, w-trigger_size:] = trigger_value.to(device)
                data = data_clone
            
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            
            for p in preds:
                if p.item() == 0:
                    ttr_count += 1
                total += 1
    return all_preds, ttr_count, total

def main():
    device = torch.device("cpu")
    batch_size = 256
    
    _, test_loader = get_cifar10_dataloaders("./data", batch_size=batch_size, use_fake=False)
    filtered_loader = get_filtered_loader(test_loader, 0, batch_size) # 9000 images
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    trigger_value = (1.0 - mean) / std
    
    mcf = SmallCNN().to(device)
    mcf.load_state_dict(torch.load("results_phase2/iid_counterfactual/final_model.pt", map_location=device))
    
    clean_model = SmallCNN().to(device)
    clean_model.load_state_dict(torch.load("results/iid_clean/final_model.pt", map_location=device))
    
    attack_model = SmallCNN().to(device)
    attack_model.load_state_dict(torch.load("results_phase2/iid_attack_logged/final_model.pt", map_location=device))
    
    mcf_clean_preds, _, _ = get_predictions_and_ttr(mcf, test_loader, device)
    mcf_trig_preds, mcf_ttr_count, mcf_ttr_total = get_predictions_and_ttr(mcf, filtered_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
    
    print(f"M_CF TTR: {mcf_ttr_count}/{mcf_ttr_total} = {100.0 * mcf_ttr_count/mcf_ttr_total:.4f}%")
    
    attack_clean_preds, _, _ = get_predictions_and_ttr(attack_model, test_loader, device)
    clean_trig_preds, _, _ = get_predictions_and_ttr(clean_model, filtered_loader, device, is_triggered=True, trigger_size=3, trigger_value=trigger_value)
    
    agree_attack_clean = sum(1 for p1, p2 in zip(attack_clean_preds, mcf_clean_preds) if p1 == p2)
    print(f"Attack Clean CF Agreement: {agree_attack_clean}/10000 = {100.0 * agree_attack_clean/10000:.4f}%")
    
    agree_clean_trig = sum(1 for p1, p2 in zip(clean_trig_preds, mcf_trig_preds) if p1 == p2)
    print(f"Phase-1 Clean Triggered CF Agreement: {agree_clean_trig}/9000 = {100.0 * agree_clean_trig/9000:.4f}%")

if __name__ == "__main__":
    main()
