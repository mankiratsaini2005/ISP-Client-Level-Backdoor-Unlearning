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
import torchvision

out_dir = "results_phase2/audit_stage1"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. Canonical Results Table
# ---------------------------------------------------------
print("Generating Canonical Results Table...")

def safe_get(d, keys, default=None):
    curr = d
    for k in keys:
        if k in curr: curr = curr[k]
        else: return default
    return curr

# Read source JSONs
with open("results/iid_clean/final_results.json") as f: iid_clean_p1 = json.load(f)
with open("results/iid_attack/final_results.json") as f: iid_attack_p1 = json.load(f)
with open("results/noniid_clean/final_results.json") as f: noniid_clean_p1 = json.load(f)
with open("results/noniid_attack/final_results.json") as f: noniid_attack_p1 = json.load(f)

with open("results_phase2/phase2_kd_evaluation.json") as f: phase2_iid = json.load(f)
with open("results_phase2/noniid_kd_evaluation.json") as f: phase2_noniid = json.load(f)
with open("results_phase2/noniid_evaluation.json") as f: phase2_noniid_old = json.load(f) # for older metrics if missing

# Merge noniid_evaluation to noniid_kd_evaluation
for k, v in phase2_noniid_old.items():
    if k not in phase2_noniid:
        phase2_noniid[k] = v

table_data = []
# Define columns
cols = ["Partition", "Model", "Clean_Accuracy", "Clean_Loss", "Triggered_Rate", "Untriggered_Airplane_Rate", "Clean_CF_Agreement", "Triggered_CF_Agreement", "Airplane_Recall", "Macro_F1", "Runtime_Seconds"]

def extract_row(partition, model_name, eval_dict, runtime_val=None):
    row = {"Partition": partition, "Model": model_name}
    row["Clean_Accuracy"] = safe_get(eval_dict, ["Clean_Acc"]) or safe_get(eval_dict, ["clean_accuracy"])
    row["Clean_Loss"] = safe_get(eval_dict, ["Clean_Loss"]) or safe_get(eval_dict, ["clean_loss"])
    row["Triggered_Rate"] = safe_get(eval_dict, ["Triggered_Rate"]) or safe_get(eval_dict, ["triggered_rate"])
    row["Untriggered_Airplane_Rate"] = safe_get(eval_dict, ["Untriggered_Airplane_Rate"]) or safe_get(eval_dict, ["untriggered_airplane_rate"])
    
    c_cf = safe_get(eval_dict, ["CF_Agreement_Clean"])
    t_cf = safe_get(eval_dict, ["CF_Agreement_Triggered"])
    if "M_CF" in model_name:
        row["Clean_CF_Agreement"] = "Reference"
        row["Triggered_CF_Agreement"] = "Reference"
    else:
        row["Clean_CF_Agreement"] = c_cf
        row["Triggered_CF_Agreement"] = t_cf
        
    row["Airplane_Recall"] = safe_get(eval_dict, ["Airplane_Recall"])
    if not row["Airplane_Recall"] and "Per_Class_Clean_Accuracy" in eval_dict:
        row["Airplane_Recall"] = eval_dict["Per_Class_Clean_Accuracy"].get("0", None)
        
    row["Macro_F1"] = safe_get(eval_dict, ["Macro_F1"]) or safe_get(eval_dict, ["clean_macro_f1"])
    row["Runtime_Seconds"] = runtime_val
    return row

def load_rt(path, key=None):
    if not os.path.exists(path): return None
    with open(path) as f:
        d = json.load(f)
        if key: return d.get(key)
        return d.get("runtime_seconds") or d.get("subtraction_time_seconds")

iid_kd_rt = load_rt("results_phase2/iid_kd/metadata.json", "kd_time_seconds")
noniid_kd_rt = load_rt("results_phase2/noniid_kd/metadata.json", "kd_time_seconds")

# Append rows
table_data.append(extract_row("IID", "Phase-1 Clean", iid_clean_p1, load_rt("results/iid_clean/final_results.json", "runtime_seconds")))
table_data.append(extract_row("IID", "Attack (No unlearning)", phase2_iid["Logged IID Attack"], load_rt("results_phase2/iid_attack_logged/runtime.json")))
table_data.append(extract_row("IID", "M_sub", phase2_iid["M_sub"], load_rt("results_phase2/iid_subtraction/subtraction_runtime.json")))
table_data.append(extract_row("IID", "M_KD", phase2_iid["M_KD"], iid_kd_rt))
table_data.append(extract_row("IID", "M_CF", phase2_iid["M_CF"], load_rt("results_phase2/iid_counterfactual/final_results.json", "runtime_seconds")))

table_data.append(extract_row("Non-IID", "Phase-1 Clean", noniid_clean_p1, load_rt("results/noniid_clean/final_results.json", "runtime_seconds")))
table_data.append(extract_row("Non-IID", "Attack (No unlearning)", phase2_noniid["Logged Non-IID Attack"], load_rt("results_phase2/noniid_attack_logged/runtime.json")))
table_data.append(extract_row("Non-IID", "M_sub", phase2_noniid["Non-IID M_sub"], load_rt("results_phase2/noniid_subtraction/subtraction_runtime.json")))
table_data.append(extract_row("Non-IID", "M_KD", phase2_noniid["Non-IID M_KD"], noniid_kd_rt))
table_data.append(extract_row("Non-IID", "M_CF", phase2_noniid["Non-IID M_CF"], load_rt("results_phase2/noniid_counterfactual/final_results.json", "runtime_seconds")))

df = pd.DataFrame(table_data)
df.to_csv(f"{out_dir}/canonical_results.csv", index=False)

# Format for MD
def fmt(v, is_pct=False):
    if v is None: return "N/A"
    if v == "Reference": return v
    if isinstance(v, (int, float)):
        if is_pct: return f"{v:.2f}%"
        return f"{v:.4f}"
    return str(v)

md_lines = ["| Partition | Model | Clean Acc | Clean Loss | ASR/TTR | UTR | Clean CF Agree | Triggered CF Agree | Air. Recall | Macro F1 | Runtime (s) |", 
            "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    line = f"| {r['Partition']} | {r['Model']} | {fmt(r['Clean_Accuracy'], True)} | {fmt(r['Clean_Loss'])} | {fmt(r['Triggered_Rate'], True)} | {fmt(r['Untriggered_Airplane_Rate'], True)} | {fmt(r['Clean_CF_Agreement'], True)} | {fmt(r['Triggered_CF_Agreement'], True)} | {fmt(r['Airplane_Recall'], True)} | {fmt(r['Macro_F1'])} | {fmt(r['Runtime_Seconds'])} |"
    md_lines.append(line)

with open(f"{out_dir}/canonical_results.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")

# ---------------------------------------------------------
# 3. Verify Non-IID M_sub Collapse
# ---------------------------------------------------------
print("Verifying Non-IID M_sub Collapse...")
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
            if p.item() == t.item():
                correct_counts[t.item()] += 1

per_class_acc = {str(i): (correct_counts[i] / class_totals[i]) * 100 for i in range(10)}

# Dump to CSV
with open(f"{out_dir}/noniid_m_sub_per_class.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Clean_Accuracy"])
    for i in range(10): writer.writerow([i, per_class_acc[str(i)]])

with open(f"{out_dir}/noniid_m_sub_predicted_distribution.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Predicted_Class", "Count", "Percentage"])
    for i in range(10): writer.writerow([i, p_counts[i], p_counts[i]/10000.0 * 100])

# Plots
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

# ---------------------------------------------------------
# 4. Verify ASR-vs-round Trajectory
# ---------------------------------------------------------
print("Verifying ASR-vs-round trajectory...")
iid_csv = pd.read_csv("results_phase2/iid_attack_logged/metrics_per_round.csv")
niid_csv = pd.read_csv("results_phase2/noniid_attack_logged/metrics_per_round.csv")

# Extract columns correctly
iid_round = iid_csv["Round"]
iid_asr = iid_csv["Triggered_Target_Rate"]
niid_round = niid_csv["Round"]
niid_asr = niid_csv["Triggered_Target_Rate"]

asr_df = pd.DataFrame({"Round": iid_round, "IID_ASR": iid_asr, "NonIID_ASR": niid_asr})
asr_df.to_csv(f"{out_dir}/asr_vs_round.csv", index=False)

plt.figure(figsize=(10,6))
plt.plot(iid_round, iid_asr, label="IID Logged Attack", marker='o')
plt.plot(niid_round, niid_asr, label="Non-IID Logged Attack", marker='s')
plt.axvline(x=30, color='r', linestyle='--', label='Attack Onset (Round 30)')
plt.xlabel("Round")
plt.ylabel("Triggered Target Rate (ASR) %")
plt.title("ASR vs Round Trajectory")
plt.legend()
plt.savefig(f"{out_dir}/asr_vs_round.png")
plt.close()

# ---------------------------------------------------------
# 5. Runtime and Storage
# ---------------------------------------------------------
print("Generating runtime and storage inventory...")
storage_rows = [
    ["Component", "Scope", "Time_Seconds"],
    ["Phase-1 IID Attack", "Full Training", load_rt("results/iid_attack/final_results.json", "runtime_seconds")],
    ["Phase-2 IID Logged Attack", "Full Training + Log", load_rt("results_phase2/iid_attack_logged/runtime.json")],
    ["Phase-2 IID M_CF", "Retraining", load_rt("results_phase2/iid_counterfactual/final_results.json", "runtime_seconds")],
    ["Phase-2 IID Subtraction", "Delta subtraction", load_rt("results_phase2/iid_subtraction/subtraction_runtime.json")],
    ["Phase-2 IID KD", "Knowledge Distillation", iid_kd_rt],
    ["Phase-1 NonIID Attack", "Full Training", load_rt("results/noniid_attack/final_results.json", "runtime_seconds")],
    ["Phase-2 NonIID Logged Attack", "Full Training + Log", load_rt("results_phase2/noniid_attack_logged/runtime.json")],
    ["Phase-2 NonIID M_CF", "Retraining", load_rt("results_phase2/noniid_counterfactual/final_results.json", "runtime_seconds")],
    ["Phase-2 NonIID Subtraction", "Delta subtraction", load_rt("results_phase2/noniid_subtraction/subtraction_runtime.json")],
    ["Phase-2 NonIID KD", "Knowledge Distillation", noniid_kd_rt],
]

with open(f"{out_dir}/runtime_storage_inventory.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for row in storage_rows: writer.writerow(row)

# ---------------------------------------------------------
# 7. Verify Historical Evaluations
# ---------------------------------------------------------
print("Comparing Non-IID Historical Evaluations...")
with open("results_phase2/noniid_evaluation.json", "r") as f:
    old_e = json.load(f)
with open("results_phase2/noniid_kd_evaluation.json", "r") as f:
    new_e = json.load(f)

mismatch = False
diff_lines = ["Historical Evaluation Verification:"]
for model in ["Logged Non-IID Attack", "Non-IID M_sub", "Non-IID M_CF"]:
    for metric in ["Clean_Acc", "Clean_Loss", "Macro_F1", "Triggered_Rate", "Untriggered_Airplane_Rate", "Airplane_Recall", "CF_Agreement_Clean", "CF_Agreement_Triggered"]:
        if metric not in old_e[model] or metric not in new_e[model]:
            diff_lines.append(f"FAIL: Missing metric {metric} in {model}")
            mismatch = True
            continue
        v_old = old_e[model][metric]
        v_new = new_e[model][metric]
        diff = abs(v_old - v_new)
        if diff > 1e-4:
            diff_lines.append(f"FAIL: {model} - {metric} | Old: {v_old} | New: {v_new} | Diff: {diff}")
            mismatch = True

if not mismatch: diff_lines.append("PASS: All metrics match within 1e-4 tolerance.")
with open(f"{out_dir}/historical_eval_diff.txt", "w") as f:
    f.write("\n".join(diff_lines))

print("Audit script finished.")
