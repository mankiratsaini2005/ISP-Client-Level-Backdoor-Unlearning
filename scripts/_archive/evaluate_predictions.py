import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.getcwd())
from src.models.small_cnn import SmallCNN

def get_model(path):
    model = SmallCNN()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

base_dir = "results/stage_6d_temporal"
early_model = get_model(os.path.join(base_dir, "early/attack/checkpoints/global_round_100.pt"))
middle_model = get_model(os.path.join(base_dir, "middle/attack/checkpoints/global_round_100.pt"))
late_model = get_model(os.path.join(base_dir, "late/attack/checkpoints/global_round_100.pt"))
no_attack_model = get_model(os.path.join(base_dir, "no_attack/checkpoints/global_round_100.pt"))
m_cf_model = get_model(os.path.join(base_dir, "matched_counterfactual/checkpoints/global_round_100.pt"))
stage4_model = get_model("results/full_retraining_final_20260909_024627/checkpoints/global_round_100.pt")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

def get_predictions(model):
    preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            preds.append(outputs.argmax(dim=1))
    return torch.cat(preds)

preds_early = get_predictions(early_model)
preds_middle = get_predictions(middle_model)
preds_late = get_predictions(late_model)
preds_no_attack = get_predictions(no_attack_model)
preds_m_cf = get_predictions(m_cf_model)
preds_stage4 = get_predictions(stage4_model)

def calc_agreement(p1, p2):
    return (p1 == p2).float().mean().item() * 100

print(f"EARLY vs NO_ATTACK Agreement: {calc_agreement(preds_early, preds_no_attack):.2f}%")
print(f"MIDDLE vs NO_ATTACK Agreement: {calc_agreement(preds_middle, preds_no_attack):.2f}%")
print(f"LATE vs NO_ATTACK Agreement: {calc_agreement(preds_late, preds_no_attack):.2f}%")
print(f"M_CF vs Stage 4 Agreement (Sanity Check): {calc_agreement(preds_m_cf, preds_stage4):.2f}%")
