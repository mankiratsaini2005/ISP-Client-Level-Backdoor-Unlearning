import pandas as pd
import hashlib

def check_csv_diff(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    diff_acc = (df1['Clean_Acc'] - df2['Clean_Acc']).abs().max()
    diff_ttr = (df1['Triggered_Target_Rate'] - df2['Triggered_Target_Rate']).abs().max()
    diff_utr = (df1['Untriggered_Airplane_Rate'] - df2['Untriggered_Airplane_Rate']).abs().max()
    
    return diff_acc, diff_ttr, diff_utr

diffs = check_csv_diff("results/iid_attack/metrics_per_round.csv", "results_phase2/iid_attack_logged/metrics_per_round.csv")
print("=== CSV METRICS DIFFERENCES (All 50 Rounds) ===")
print(f"Max Diff - Clean Accuracy: {diffs[0]:.6f}")
print(f"Max Diff - Triggered Target Rate: {diffs[1]:.6f}")
print(f"Max Diff - Untriggered Airplane Rate: {diffs[2]:.6f}")
if diffs[0] == 0 and diffs[1] == 0 and diffs[2] == 0:
    print("MATCH: EXACT")
else:
    print("MATCH: NOT EXACT")

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

print("\n=== CHECKSUMS / MATCHES ===")
files_to_check = ['initial_model.pt', 'client_partitions.pt', 'client_class_histograms.json']
for f in files_to_check:
    path1 = f"results/iid_attack/{f}"
    path2 = f"results_phase2/iid_attack_logged/{f}"
    
    h1 = hash_file(path1)
    h2 = hash_file(path2)
    
    if h1 == h2:
        print(f"{f}: HASH MATCH")
    else:
        print(f"{f}: HASH MISMATCH")
        
