import os
import json
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import glob

print("--- 1. Fix clean_cf.png ---")
with open("results_phase2/phase2_kd_evaluation.json", "r") as f:
    iid_e = json.load(f)
with open("results_phase2/noniid_kd_evaluation.json", "r") as f:
    niid_e = json.load(f)
with open("results_phase2/noniid_evaluation.json", "r") as f:
    old_niid = json.load(f)
for k, v in old_niid.items():
    if k not in niid_e: niid_e[k] = v

models_iid = ["Phase-1 Clean", "Logged IID Attack", "M_sub", "M_KD"]
models_niid = ["Phase-1 Non-IID Clean", "Logged Non-IID Attack", "Non-IID M_sub", "Non-IID M_KD"]
labels = ["Phase-1 Clean", "Attack", "M_sub", "M_KD"]

def extract(eval_dict, keys, metric):
    return [eval_dict[k][metric] for k in keys]

iid_cf = extract(iid_e, models_iid, "CF_Agreement_Clean")
niid_cf = extract(niid_e, models_niid, "CF_Agreement_Clean")

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, iid_cf, width, label='IID', color='#1f77b4')
ax.bar(x + width/2, niid_cf, width, label='Non-IID', color='#ff7f0e')
ax.set_ylabel('Agreement (%)')
ax.set_title('Clean Counterfactual Agreement')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig("results_phase2/charts/clean_cf.png")
plt.close()
print("Fixed clean_cf.png (removed 100% horizontal line)")

print("--- 2. Expand sha256_manifest.txt ---")
def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

files_to_hash = [
    # Phase-1 Checkpoints and configs
    "results/iid_clean/initial_model.pt",
    "results/iid_clean/final_model.pt",
    "results/iid_clean/client_partitions.pt",
    "results/iid_clean/metrics_per_round.csv",
    "results/iid_attack/initial_model.pt",
    "results/iid_attack/final_model.pt",
    "results/iid_attack/client_partitions.pt",
    "results/iid_attack/metrics_per_round.csv",
    "results/noniid_clean/initial_model.pt",
    "results/noniid_clean/final_model.pt",
    "results/noniid_clean/client_partitions.pt",
    "results/noniid_clean/metrics_per_round.csv",
    "results/noniid_attack/initial_model.pt",
    "results/noniid_attack/final_model.pt",
    "results/noniid_attack/client_partitions.pt",
    "results/noniid_attack/metrics_per_round.csv",
    
    # Phase-2 Checkpoints, Partitions, and configs
    "results_phase2/iid_attack_logged/initial_model.pt",
    "results_phase2/iid_attack_logged/final_model.pt",
    "results_phase2/iid_attack_logged/client_partitions.pt",
    "results_phase2/iid_attack_logged/metrics_per_round.csv",
    "results_phase2/iid_attack_logged/runtime.json",
    "results_phase2/iid_subtraction/final_model.pt",
    "results_phase2/iid_subtraction/subtraction_runtime.json",
    "results_phase2/iid_kd/final_model.pt",
    "results_phase2/iid_kd/metadata.json",
    "results_phase2/iid_counterfactual/initial_model.pt",
    "results_phase2/iid_counterfactual/final_model.pt",
    "results_phase2/iid_counterfactual/metrics_per_round.csv",
    "results_phase2/iid_counterfactual/config.json",
    "results_phase2/iid_counterfactual/final_results.json",
    "results_phase2/noniid_attack_logged/initial_model.pt",
    "results_phase2/noniid_attack_logged/final_model.pt",
    "results_phase2/noniid_attack_logged/client_partitions.pt",
    "results_phase2/noniid_attack_logged/client_class_histograms.json",
    "results_phase2/noniid_attack_logged/metrics_per_round.csv",
    "results_phase2/noniid_attack_logged/runtime.json",
    "results_phase2/noniid_subtraction/final_model.pt",
    "results_phase2/noniid_subtraction/subtraction_runtime.json",
    "results_phase2/noniid_kd/final_model.pt",
    "results_phase2/noniid_kd/metadata.json",
    "results_phase2/noniid_counterfactual/initial_model.pt",
    "results_phase2/noniid_counterfactual/final_model.pt",
    "results_phase2/noniid_counterfactual/config.json",
    "results_phase2/noniid_counterfactual/metrics_per_round.csv",
    "results_phase2/noniid_counterfactual/final_results.json",
    
    # Phase-2 Result/Eval JSONs
    "results_phase2/phase2_evaluation_superseded.json",
    "results_phase2/phase2_evaluation_corrected.json",
    "results_phase2/phase2_kd_evaluation.json",
    "results_phase2/noniid_evaluation.json",
    "results_phase2/noniid_kd_evaluation.json",
    
    # Proxy Indices
    "results_phase2/proxy_data/cifar100_proxy_indices.json"
]

manifest_lines = []
missing_files = []
hashed_count = 0
for f in files_to_hash:
    if os.path.exists(f):
        manifest_lines.append(f"{hash_file(f)}  {f}")
        hashed_count += 1
    else:
        missing_files.append(f)
        manifest_lines.append(f"MISSING: {f}")

with open("sha256_manifest.txt", "w") as f:
    f.write("\n".join(manifest_lines) + "\n")

print(f"Total files in list: {len(files_to_hash)}")
print(f"Total hashed: {hashed_count}")
print(f"Total missing: {len(missing_files)}")
if missing_files:
    print("Missing files explicitly reported in manifest:")
    for m in missing_files: print(f" - {m}")

