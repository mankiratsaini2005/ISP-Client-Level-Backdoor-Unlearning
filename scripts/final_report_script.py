import os
import json
import numpy as np
import matplotlib.pyplot as plt

print("--- 2. Regenerate Charts from JSON ---")
with open("results_phase2/phase2_kd_evaluation.json", "r") as f:
    iid_e = json.load(f)
with open("results_phase2/noniid_kd_evaluation.json", "r") as f:
    niid_e = json.load(f)
with open("results_phase2/noniid_evaluation.json", "r") as f:
    old_niid = json.load(f)
    
# Merge old into new to get 'Phase-1 Non-IID Clean'
for k, v in old_niid.items():
    if k not in niid_e:
        niid_e[k] = v

models_iid = ["Phase-1 Clean", "Logged IID Attack", "M_sub", "M_KD", "M_CF"]
models_niid = ["Phase-1 Non-IID Clean", "Logged Non-IID Attack", "Non-IID M_sub", "Non-IID M_KD", "Non-IID M_CF"]

def extract(eval_dict, keys, metric):
    res = []
    for k in keys:
        if metric == "Airplane_Recall":
            res.append(eval_dict[k]["Per_Class_Clean_Accuracy"]["0"])
        else:
            res.append(eval_dict[k][metric])
    return res

iid_acc = extract(iid_e, models_iid, "Clean_Acc")
niid_acc = extract(niid_e, models_niid, "Clean_Acc")

iid_cf = extract(iid_e, models_iid, "CF_Agreement_Clean")
niid_cf = extract(niid_e, models_niid, "CF_Agreement_Clean")

iid_rec = extract(iid_e, models_iid, "Airplane_Recall")
niid_rec = extract(niid_e, models_niid, "Airplane_Recall")

labels = ["Phase-1 Clean", "Attack", "M_sub", "M_KD", "M_CF (Ref)"]
x = np.arange(len(labels))
width = 0.35

def plot_b(d1, d2, title, ylab, name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, d1, width, label='IID', color='#1f77b4')
    ax.bar(x + width/2, d2, width, label='Non-IID', color='#ff7f0e')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(f"results_phase2/charts/{name}")
    plt.close()

plot_b(iid_acc, niid_acc, 'Clean Accuracy Comparison', 'Accuracy (%)', 'clean_acc.png')
plot_b(iid_cf, niid_cf, 'Clean Counterfactual Agreement', 'Agreement (%)', 'clean_cf.png')
plot_b(iid_rec, niid_rec, 'Airplane Recall', 'Recall (%)', 'airplane_recall.png')

with open("results_phase2/iid_kd/metadata.json", "r") as f:
    iid_meta = json.load(f)
with open("results_phase2/noniid_kd/metadata.json", "r") as f:
    noniid_meta = json.load(f)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), iid_meta["kd_loss_history"], marker='o', label='IID KD Loss')
plt.plot(range(1, 6), noniid_meta["kd_loss_history"], marker='s', label='Non-IID KD Loss')
plt.xlabel("Epoch")
plt.ylabel("Scaled KL Divergence")
plt.title("KD Loss Curves")
plt.legend()
plt.savefig("results_phase2/charts/kd_loss.png")
plt.close()

print("Charts saved to results_phase2/charts/")
