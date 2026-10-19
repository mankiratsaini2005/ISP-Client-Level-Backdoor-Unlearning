import torch
import json
import os
import torch.nn.functional as F
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from scripts.unlearn_stage5b import create_filtered_asr_dataset

device = torch.device("cpu")

with open("results/stage7c_d1_canonical/unlearning/federaser/eval_indices.json", "r") as f:
    eval_indices = json.load(f)

_, test_loader = get_cifar10_dataloaders("./data", 64, False)
test_dataset = test_loader.dataset
eval_clean_dataset = torch.utils.data.Subset(test_dataset, eval_indices)
eval_asr_dataset = create_filtered_asr_dataset(test_dataset, eval_indices, target_class=0, trigger_size=3)

eval_clean_loader = torch.utils.data.DataLoader(eval_clean_dataset, batch_size=64, shuffle=False)
eval_asr_loader = torch.utils.data.DataLoader(eval_asr_dataset, batch_size=64, shuffle=False)

def get_preds(model_path):
    m = SmallCNN().to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    
    clean_preds = []
    clean_correct = 0
    clean_total = 0
    with torch.no_grad():
        for x, y in eval_clean_loader:
            x, y = x.to(device), y.to(device)
            out = m(x)
            pred = out.argmax(dim=1)
            clean_preds.append(pred)
            clean_correct += (pred == y).sum().item()
            clean_total += y.size(0)
            
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
        "preds": clean_preds,
        "clean_acc": 100.0 * clean_correct / clean_total,
        "asr": 100.0 * asr_correct / asr_total if asr_total > 0 else 0.0
    }

paths = {
    "D0": {
        "m_cf": "results/stage7a_c1_mcf_20260911_003109/checkpoints/global_round_100.pt",
        "m_att": "results/stage7a_c1_attacked_20260910_233012/checkpoints/global_round_100.pt",
        "m_fe": "results/stage7a_unlearning/c1/federaser/checkpoints/global_round_100.pt"
    },
    "D1": {
        "m_cf": "results/stage7c_d1_mcf_20260911_230957/checkpoints/global_round_100.pt",
        "m_att": "results/attack_badnets_20260912_182946/checkpoints/global_round_100.pt",
        "m_fe": "results/stage7c_d1_canonical/unlearning/federaser/checkpoints/global_round_100.pt"
    },
    "D2": {
        "m_cf": "results/stage7c_d2_mcf20260911_213425/checkpoints/global_round_100.pt",
        "m_att": "results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423/checkpoints/global_round_100.pt",
        "m_fe": "results/stage7c_d2/unlearning/federaser/checkpoints/global_round_100.pt"
    },
    "D3": {
        "m_cf": "results/stage7a_c3_mcf_20260911_032646/checkpoints/global_round_100.pt",
        "m_att": "results/stage7a_c3_attacked_20260911_022619/checkpoints/global_round_100.pt",
        "m_fe": "results/stage7a_unlearning/c3/federaser/checkpoints/global_round_100.pt"
    }
}

final_results = {}

for cond, p in paths.items():
    print(f"Processing {cond}...")
    try:
        res_cf = get_preds(p["m_cf"])
        res_att = get_preds(p["m_att"])
        res_fe = get_preds(p["m_fe"])
        
        att_cf_agree = (res_att["preds"] == res_cf["preds"]).float().mean().item() * 100.0
        fe_cf_agree = (res_fe["preds"] == res_cf["preds"]).float().mean().item() * 100.0
        
        final_results[cond] = {
            "ASR(M_Attacked)": res_att["asr"],
            "ASR(M_CF)": res_cf["asr"],
            "ASR_Excess": res_att["asr"] - res_cf["asr"],
            "Clean(M_Attacked)": res_att["clean_acc"],
            "Clean(M_CF)": res_cf["clean_acc"],
            "Clean_Utility_Diff": res_att["clean_acc"] - res_cf["clean_acc"],
            "Agreement(M_Attacked, M_CF)": att_cf_agree,
            "Disagreement(M_Attacked, M_CF)": 100.0 - att_cf_agree,
            "ASR(M_FE)": res_fe["asr"],
            "ASR_Reduction": res_att["asr"] - res_fe["asr"],
            "Agreement(M_FE, M_CF)": fe_cf_agree,
            "CF_Gap(M_FE)": 100.0 - fe_cf_agree,
            "Utility_Gap(M_FE)": res_cf["clean_acc"] - res_fe["clean_acc"]
        }
    except Exception as e:
        print(f"Error processing {cond}: {e}")

with open("results/stage7c_final_factorial_robustness_audit.json", "w") as f:
    json.dump(final_results, f, indent=4)
print(json.dumps(final_results, indent=4))
