import torch
import json
import os
import torch.nn.functional as F
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset

device = torch.device("cpu")

with open("results/stage7c_d1/unlearning/federaser/eval_indices.json", "r") as f:
    eval_indices = json.load(f)

_, test_loader = get_cifar10_dataloaders("./data", 64, False)
test_dataset = test_loader.dataset
eval_clean_dataset = torch.utils.data.Subset(test_dataset, eval_indices)
eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)

eval_clean_loader = torch.utils.data.DataLoader(eval_clean_dataset, batch_size=64, shuffle=False)
eval_asr_loader = torch.utils.data.DataLoader(eval_asr_dataset, batch_size=64, shuffle=False)

def get_predictions(model_path):
    m = SmallCNN().to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    
    clean_logits = []
    clean_preds = []
    clean_loss = 0
    clean_total = 0
    clean_correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in eval_clean_loader:
            x, y = x.to(device), y.to(device)
            out = m(x)
            clean_logits.append(out)
            pred = out.argmax(dim=1)
            clean_preds.append(pred)
            clean_loss += criterion(out, y).item() * x.size(0)
            clean_correct += (pred == y).sum().item()
            clean_total += y.size(0)
            
    clean_logits = torch.cat(clean_logits, dim=0)
    clean_preds = torch.cat(clean_preds, dim=0)
    
    asr_correct = 0
    asr_total = 0
    with torch.no_grad():
        for x, y in eval_asr_loader:
            x, y = x.to(device), y.to(device)
            out = m(x)
            pred = out.argmax(dim=1)
            asr_correct += (pred == y).sum().item()
            asr_total += y.size(0)
            
    return {
        "logits": clean_logits,
        "preds": clean_preds,
        "Clean Accuracy": 100.0 * clean_correct / clean_total,
        "Clean Loss": clean_loss / clean_total,
        "ASR": 100.0 * asr_correct / asr_total if asr_total > 0 else 0.0
    }

mcf = get_predictions("results/stage7c_d1_mcf_20260911_230957/checkpoints/global_round_100.pt")

models = {
    "D1 M_CF (Ref)": "results/stage7c_d1_mcf_20260911_230957/checkpoints/global_round_100.pt",
    "D1 Attacked": "results/stage7c_d1_attacked_20260911_230919/checkpoints/global_round_100.pt",
    "Subtraction": "results/stage7c_d1/unlearning/subtraction/checkpoints/subtraction_model.pt",
    "KD": "results/stage7c_d1/unlearning/kd/checkpoints/final_unlearned.pt",
    "FedEraser": "results/stage7c_d1/unlearning/federaser/checkpoints/global_round_100.pt"
}

results = []
ref_acc = mcf["Clean Accuracy"]

for name, path in models.items():
    res = get_predictions(path)
    agreement = (res["preds"] == mcf["preds"]).float().mean().item() * 100.0
    
    results.append({
        "Method": name,
        "Clean Acc": f"{res['Clean Accuracy']:.2f}%",
        "Clean Loss": f"{res['Clean Loss']:.3f}",
        "ASR": f"{res['ASR']:.2f}%",
        "CF Agreement": f"{agreement:.2f}%",
        "CF Gap": f"{(100.0 - agreement):.2f}%",
        "Utility Gap": f"{(ref_acc - res['Clean Accuracy']):.2f}%"
    })

print("\n--- FINAL EVALUATION RESULTS ---")
header = "| Method | Clean Acc | Clean Loss | ASR | CF Agreement | CF Gap | Utility Gap |"
print(header)
print("|---|---:|---:|---:|---:|---:|---:|")
for r in results:
    print(f"| {r['Method']} | {r['Clean Acc']} | {r['Clean Loss']} | {r['ASR']} | {r['CF Agreement']} | {r['CF Gap']} | {r['Utility Gap']} |")
