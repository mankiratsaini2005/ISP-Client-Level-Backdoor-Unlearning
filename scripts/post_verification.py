import pandas as pd
import hashlib
import json
import torch
import os

def check_csv_diff(clean_csv, attack_csv):
    df_clean = pd.read_csv(clean_csv)
    df_attack = pd.read_csv(attack_csv)
    
    # Filter to rounds 1-29
    df_clean = df_clean[df_clean['Round'] < 30]
    df_attack = df_attack[df_attack['Round'] < 30]
    
    diff_acc = (df_clean['Clean_Acc'] - df_attack['Clean_Acc']).abs().max()
    diff_ttr = (df_clean['Triggered_Target_Rate'] - df_attack['Triggered_Target_Rate']).abs().max()
    diff_utr = (df_clean['Untriggered_Airplane_Rate'] - df_attack['Untriggered_Airplane_Rate']).abs().max()
    
    return diff_acc, diff_ttr, diff_utr

print("=== CSV DIFFERENCES (Rounds 1-29) ===")
iid_diff = check_csv_diff("results/iid_clean/metrics_per_round.csv", "results/iid_attack/metrics_per_round.csv")
print(f"IID: Max Diff - Clean_Acc: {iid_diff[0]:.6f}, TTR: {iid_diff[1]:.6f}, UTR: {iid_diff[2]:.6f}")

noniid_diff = check_csv_diff("results/noniid_clean/metrics_per_round.csv", "results/noniid_attack/metrics_per_round.csv")
print(f"Non-IID: Max Diff - Clean_Acc: {noniid_diff[0]:.6f}, TTR: {noniid_diff[1]:.6f}, UTR: {noniid_diff[2]:.6f}")

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def compare_pt_tensors(file1, file2):
    t1 = torch.load(file1, map_location='cpu')
    t2 = torch.load(file2, map_location='cpu')
    
    if type(t1) == list and type(t2) == list:
        # Partitions
        for i in range(len(t1)):
            if t1[i].indices != t2[i].indices:
                return False
        return True
    elif type(t1) == dict and type(t2) == dict:
        # State dicts
        for k in t1.keys():
            if not torch.equal(t1[k], t2[k]):
                return False
        return True
    return False

print("\n=== CHECKSUMS / MATCHES ===")
files_to_check = ['initial_model.pt', 'client_partitions.pt', 'client_class_histograms.json']
for mode in ['iid', 'noniid']:
    for f in files_to_check:
        path_clean = f"results/{mode}_clean/{f}"
        path_attack = f"results/{mode}_attack/{f}"
        
        h1 = hash_file(path_clean)
        h2 = hash_file(path_attack)
        
        if h1 == h2:
            print(f"{mode} {f}: HASH MATCH")
        else:
            print(f"{mode} {f}: HASH MISMATCH. Checking tensor content...")
            if f.endswith(".pt"):
                tensor_match = compare_pt_tensors(path_clean, path_attack)
                print(f"  -> Tensor Match: {tensor_match}")
            else:
                print(f"  -> Direct file compare required.")
                
