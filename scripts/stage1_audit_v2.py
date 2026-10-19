import json
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from src.models.small_cnn import SmallCNN
import src.data.cifar10

out_dir = "results_phase2/audit_stage1"
os.makedirs(out_dir, exist_ok=True)

print("--- 1. Canonical Results Table ---")

def load_json(path):
    if not os.path.exists(path): return {}
    with open(path) as f: return json.load(f)

iid_clean_p1 = load_json("results/iid_clean/final_results.json")
iid_attack_p1 = load_json("results/iid_attack/final_results.json")
noniid_clean_p1 = load_json("results/noniid_clean/final_results.json")
noniid_attack_p1 = load_json("results/noniid_attack/final_results.json")

iid_eval = load_json("results_phase2/phase2_kd_evaluation.json")
noniid_eval_old = load_json("results_phase2/noniid_evaluation.json")
noniid_eval_new = load_json("results_phase2/noniid_kd_evaluation.json")

# Extract runtime helpers
def get_rt(path, key="runtime_seconds"):
    d = load_json(path)
    val = d.get(key)
    if val is None: val = d.get("subtraction_time_seconds")
    return val

table_data = []

def create_row(partition, model, base_dict, eval_dict, rt_label, rt_val):
    # base_dict: dictionary containing actual performance (for Phase-1 clean this is the original JSON)
    # eval_dict: dictionary containing CF Agreements, Per_Class_Clean_Accuracy (for Phase-1 clean this is the later eval JSON)
    row = {"Partition": partition, "Model": model}
    
    c_acc = base_dict.get("Clean_Acc") if base_dict.get("Clean_Acc") is not None else base_dict.get("clean_accuracy")
    c_loss = base_dict.get("Clean_Loss") if base_dict.get("Clean_Loss") is not None else base_dict.get("clean_loss")
    t_rate = base_dict.get("Triggered_Rate") if base_dict.get("Triggered_Rate") is not None else base_dict.get("triggered_rate")
    utr = base_dict.get("Untriggered_Airplane_Rate") if base_dict.get("Untriggered_Airplane_Rate") is not None else base_dict.get("untriggered_airplane_rate")
    f1 = base_dict.get("Macro_F1") if base_dict.get("Macro_F1") is not None else base_dict.get("clean_macro_f1")
    
    row["Clean_Accuracy"] = c_acc
    row["Clean_Loss"] = c_loss
    row["Triggered_Rate"] = t_rate
    row["Untriggered_Airplane_Rate"] = utr
    row["Macro_F1"] = f1
    
    # CF Agreements
    if "M_CF" in model:
        row["Clean_CF_Agreement"] = "Reference"
        row["Triggered_CF_Agreement"] = "Reference"
    else:
        c_cf = eval_dict.get("CF_Agreement_Clean")
        t_cf = eval_dict.get("CF_Agreement_Triggered")
        row["Clean_CF_Agreement"] = c_cf if c_cf is not None else "N/A"
        row["Triggered_CF_Agreement"] = t_cf if t_cf is not None else "N/A"
        
    # Airplane Recall
    ar = eval_dict.get("Airplane_Recall")
    if ar is None and "Per_Class_Clean_Accuracy" in eval_dict:
        ar = eval_dict["Per_Class_Clean_Accuracy"].get("0")
    row["Airplane_Recall"] = ar if ar is not None else "N/A"
    
    # Runtime
    if rt_val is not None:
        row["Runtime_Seconds"] = f"{rt_label} ({rt_val:.2f}s)"
    else:
        row["Runtime_Seconds"] = "N/A"
        
    return row

# IID Rows
table_data.append(create_row("IID", "Phase-1 Clean", iid_clean_p1, iid_eval.get("Phase-1 Clean", {}), "Full Training", get_rt("results/iid_clean/final_results.json")))
table_data.append(create_row("IID", "Attack (No unlearning)", iid_eval.get("Logged IID Attack", {}), iid_eval.get("Logged IID Attack", {}), "Full Training + Log", get_rt("results_phase2/iid_attack_logged/runtime.json")))
table_data.append(create_row("IID", "M_sub", iid_eval.get("M_sub", {}), iid_eval.get("M_sub", {}), "Subtraction-only", get_rt("results_phase2/iid_subtraction/subtraction_runtime.json")))
# KD time = subtraction + kd
iid_sub_rt = get_rt("results_phase2/iid_subtraction/subtraction_runtime.json")
iid_kd_rt = get_rt("results_phase2/iid_kd/metadata.json", "kd_time_seconds")
iid_tot_kd = (iid_sub_rt + iid_kd_rt) if (iid_sub_rt is not None and iid_kd_rt is not None) else None
table_data.append(create_row("IID", "M_KD", iid_eval.get("M_KD", {}), iid_eval.get("M_KD", {}), "Subtraction+KD", iid_tot_kd))
table_data.append(create_row("IID", "M_CF", iid_eval.get("M_CF", {}), iid_eval.get("M_CF", {}), "Retraining", get_rt("results_phase2/iid_counterfactual/final_results.json")))

# Non-IID Rows
table_data.append(create_row("Non-IID", "Phase-1 Clean", noniid_clean_p1, noniid_eval_old.get("Phase-1 Non-IID Clean", {}), "Full Training", get_rt("results/noniid_clean/final_results.json")))
table_data.append(create_row("Non-IID", "Attack (No unlearning)", noniid_eval_new.get("Logged Non-IID Attack", {}), noniid_eval_new.get("Logged Non-IID Attack", {}), "Full Training + Log", get_rt("results_phase2/noniid_attack_logged/runtime.json")))
table_data.append(create_row("Non-IID", "M_sub", noniid_eval_new.get("Non-IID M_sub", {}), noniid_eval_new.get("Non-IID M_sub", {}), "Subtraction-only", get_rt("results_phase2/noniid_subtraction/subtraction_runtime.json")))
niid_sub_rt = get_rt("results_phase2/noniid_subtraction/subtraction_runtime.json")
niid_kd_rt = get_rt("results_phase2/noniid_kd/metadata.json", "kd_time_seconds")
niid_tot_kd = (niid_sub_rt + niid_kd_rt) if (niid_sub_rt is not None and niid_kd_rt is not None) else None
table_data.append(create_row("Non-IID", "M_KD", noniid_eval_new.get("Non-IID M_KD", {}), noniid_eval_new.get("Non-IID M_KD", {}), "Subtraction+KD", niid_tot_kd))
table_data.append(create_row("Non-IID", "M_CF", noniid_eval_new.get("Non-IID M_CF", {}), noniid_eval_new.get("Non-IID M_CF", {}), "Retraining", get_rt("results_phase2/noniid_counterfactual/final_results.json")))

df = pd.DataFrame(table_data)
df.to_csv(f"{out_dir}/canonical_results.csv", index=False)

def fmt(v, is_pct=False):
    if v is None: return "N/A"
    if v == "Reference" or v == "N/A": return v
    if isinstance(v, (int, float)):
        if is_pct: return f"{v:.2f}%"
        return f"{v:.4f}"
    return str(v)

md_lines = ["| Partition | Model | Clean Acc | Clean Loss | ASR/TTR | UTR | Clean CF Agree | Triggered CF Agree | Air. Recall | Macro F1 | Runtime |", 
            "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    line = f"| {r['Partition']} | {r['Model']} | {fmt(r['Clean_Accuracy'], True)} | {fmt(r['Clean_Loss'])} | {fmt(r['Triggered_Rate'], True)} | {fmt(r['Untriggered_Airplane_Rate'], True)} | {fmt(r['Clean_CF_Agreement'], True)} | {fmt(r['Triggered_CF_Agreement'], True)} | {fmt(r['Airplane_Recall'], True)} | {fmt(r['Macro_F1'])} | {r['Runtime_Seconds']} |"
    md_lines.append(line)
with open(f"{out_dir}/canonical_results.md", "w") as f: f.write("\n".join(md_lines) + "\n")

print("--- 2. Storage Inventory ---")
paths_to_check = [
    ("results_phase2/iid_attack_logged/trajectory", "Logged IID Attack Trajectories"),
    ("results_phase2/noniid_attack_logged/trajectory", "Logged Non-IID Attack Trajectories"),
    ("results_phase2/iid_attack_logged/client_partitions.pt", "IID Client Partitions"),
    ("results_phase2/noniid_attack_logged/client_partitions.pt", "Non-IID Client Partitions"),
    ("results_phase2/iid_subtraction/final_model.pt", "IID M_sub Checkpoint"),
    ("results_phase2/noniid_subtraction/final_model.pt", "Non-IID M_sub Checkpoint"),
]
def get_size(path):
    if not os.path.exists(path): return 0, 0
    if os.path.isfile(path): return os.path.getsize(path), 1
    total_size, file_count = 0, 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
                file_count += 1
    return total_size, file_count

storage_data = []
for p, desc in paths_to_check:
    size, count = get_size(p)
    exists = os.path.exists(p)
    mb = size / (1024**2)
    gb = size / (1024**3)
    storage_data.append({
        "Description": desc, "Path": p, "Exists": exists, "File_Count": count, 
        "Size_Bytes": size, "Size_MiB": f"{mb:.2f}", "Size_GiB": f"{gb:.2f}"
    })
pd.DataFrame(storage_data).to_csv(f"{out_dir}/artifact_storage_inventory.csv", index=False)

print("--- 3. Historical Evaluation Verification ---")
mismatch = False
diff_lines = ["Historical Evaluation Verification:"]
for model in ["Logged Non-IID Attack", "Non-IID M_sub", "Non-IID M_CF"]:
    for metric in ["Clean_Acc", "Clean_Loss", "Macro_F1", "Triggered_Rate", "Untriggered_Airplane_Rate", "Airplane_Recall", "CF_Agreement_Clean", "CF_Agreement_Triggered"]:
        v_old = noniid_eval_old.get(model, {}).get(metric)
        v_new = noniid_eval_new.get(model, {}).get(metric)
        
        if v_old is None or v_new is None or not np.isfinite(v_old) or not np.isfinite(v_new):
            diff_lines.append(f"FAIL: Missing or invalid metric {metric} in {model}. Old: {v_old}, New: {v_new}")
            mismatch = True
            continue
            
        diff = abs(v_old - v_new)
        diff_lines.append(f"CHECK: {model} - {metric} | Old: {v_old} | New: {v_new} | Diff: {diff}")
        if diff > 1e-4:
            diff_lines.append(f"-> FAIL: Difference exceeds 1e-4")
            mismatch = True

if not mismatch:
    diff_lines.append("FINAL PASS")
    print("Historical Eval Verification: FINAL PASS")
else:
    diff_lines.append("FINAL FAIL")
    print("Historical Eval Verification: FINAL FAIL")
with open(f"{out_dir}/historical_eval_diff.txt", "w") as f:
    f.write("\n".join(diff_lines) + "\n")

print("--- 4. Validate ASR Trajectories ---")
iid_csv = pd.read_csv("results_phase2/iid_attack_logged/metrics_per_round.csv")
niid_csv = pd.read_csv("results_phase2/noniid_attack_logged/metrics_per_round.csv")
# Join explicitly on Round
merged_csv = pd.merge(iid_csv, niid_csv, on="Round", suffixes=('_IID', '_NonIID'))
merged_csv.to_csv(f"{out_dir}/asr_vs_round.csv", index=False)

iid_42 = iid_csv[iid_csv['Round'] == 42]['Triggered_Target_Rate'].values[0]
iid_50 = iid_csv[iid_csv['Round'] == 50]['Triggered_Target_Rate'].values[0]
print(f"IID ASR at Round 42: {iid_42:.2f}%")
print(f"IID ASR at Round 50: {iid_50:.2f}%")
change = iid_50 - iid_42
print(f"Change between Round 42 and 50: {change:.2f}%")
if change > 0:
    print("IID ASR was indeed increasing late in training.")

plt.figure(figsize=(10,6))
plt.plot(merged_csv["Round"], merged_csv["Triggered_Target_Rate_IID"], label="IID Logged Attack", marker='o')
plt.plot(merged_csv["Round"], merged_csv["Triggered_Target_Rate_NonIID"], label="Non-IID Logged Attack", marker='s')
plt.axvline(x=30, color='r', linestyle='--', label='Attack Onset (Round 30)')
plt.xlabel("Round")
plt.ylabel("Triggered Target Rate (ASR) %")
plt.title("ASR vs Round Trajectory")
plt.legend()
plt.savefig(f"{out_dir}/asr_vs_round.png")
plt.close()

print("--- 5. Validate Canonical Rows / M_sub Collapse ---")
device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
_, test_loader = src.data.cifar10.get_cifar10_dataloaders("./data", batch_size=256, use_fake=False)
m = SmallCNN().to(device)
m.load_state_dict(torch.load("results_phase2/noniid_subtraction/final_model.pt", map_location=device))
m.eval()
p_counts = {i: 0 for i in range(10)}
correct_counts = {i: 0 for i in range(10)}
class_totals = {i: 0 for i in range(10)}
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        preds = m(data).argmax(dim=1)
        for p, t in zip(preds, target):
            p_counts[p.item()] += 1
            class_totals[t.item()] += 1
            if p.item() == t.item(): correct_counts[t.item()] += 1
per_class_acc = {str(i): (correct_counts[i] / class_totals[i]) * 100 for i in range(10)}

with open(f"{out_dir}/noniid_m_sub_per_class.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Clean_Accuracy"])
    for i in range(10): writer.writerow([i, per_class_acc[str(i)]])
with open(f"{out_dir}/noniid_m_sub_predicted_distribution.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Predicted_Class", "Count", "Percentage"])
    for i in range(10): writer.writerow([i, p_counts[i], p_counts[i]/10000.0 * 100])

print(f"Total predictions across all classes: {sum(p_counts.values())}")

classes = [str(i) for i in range(10)]
plt.figure(figsize=(8,4))
plt.bar(classes, [per_class_acc[c] for c in classes], color='red')
plt.title("Non-IID M_sub Per-Class Clean Accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy (%)")
plt.savefig(f"{out_dir}/noniid_m_sub_per_class_accuracy.png")
plt.close()
plt.figure(figsize=(8,4))
plt.bar(classes, [p_counts[i] for i in range(10)], color='orange')
plt.title("Non-IID M_sub Predicted-Class Distribution (10k test)")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.savefig(f"{out_dir}/noniid_m_sub_predicted_distribution.png")
plt.close()
