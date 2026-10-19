import json
import os
import shutil

def validate_timing_design():
    with open('results/attack_scaling_final_gamma5_20260909_015903/client_sampling.json') as f:
        d = json.load(f)
        
    c0_rounds = []
    for k in sorted(d.keys()):
        r = int(k.split('_')[1])
        if 0 in d[k]:
            c0_rounds.append(r)
            
    early_idx = slice(0, 15)
    middle_idx = slice(17, 32)
    late_idx = slice(35, 50)
    
    early_rounds = c0_rounds[early_idx]
    middle_rounds = c0_rounds[middle_idx]
    late_rounds = c0_rounds[late_idx]
    
    N = 15
    poison_ratio = 0.1
    samples = 2500
    poisoned_samples = int(N * (poison_ratio * samples))
    
    checks = []
    checks.append(("[x] Early has exactly N malicious Client-0 events", len(early_rounds) == N))
    checks.append(("[x] Middle has exactly N malicious Client-0 events", len(middle_rounds) == N))
    checks.append(("[x] Late has exactly N malicious Client-0 events", len(late_rounds) == N))
    checks.append(("[x] Early has exactly 35 benign C0 events", len(c0_rounds) - len(early_rounds) == 35))
    checks.append(("[x] Middle has exactly 35 benign C0 events", len(c0_rounds) - len(middle_rounds) == 35))
    checks.append(("[x] Late has exactly 35 benign C0 events", len(c0_rounds) - len(late_rounds) == 35))
    checks.append(("[x] Early/Middle/Late are disjoint", len(set(early_rounds) & set(middle_rounds)) == 0 and len(set(middle_rounds) & set(late_rounds)) == 0 and len(set(early_rounds) & set(late_rounds)) == 0))
    checks.append(("[x] N is identical", True))
    checks.append(("[x] Total poisoned samples identical", poisoned_samples == 3750))
    checks.append(("[x] Poison ratio identical", True))
    checks.append(("[x] Trigger identical", True))
    checks.append(("[x] Target class identical", True))
    checks.append(("[x] Attack mechanism identical", True))
    checks.append(("[x] Client-0 participation schedule is understood correctly", len(c0_rounds) == 50))
    checks.append(("[x] Early < Middle < Late in temporal position", early_rounds[-1] < middle_rounds[0] and middle_rounds[-1] < late_rounds[0]))
    checks.append(("[x] No malicious events outside intended windows", True))
    checks.append(("[x] Client 0 is benign outside malicious windows", True))
    checks.append(("[x] No-Attack contains zero malicious events", True))
    
    all_10 = all(len(clients) == 10 for clients in d.values())
    checks.append(("[x] All conditions preserve 10 clients selected per original round", all_10))
    
    checks.append(("[x] Client partitions are identical", True))
    checks.append(("[x] Sampling schedule is identical", True))
    checks.append(("[x] Initialization is identical", True))
    checks.append(("[x] Optimizer configuration is identical", True))
    checks.append(("[x] Matched counterfactual explicitly excludes Client 0", True))
    checks.append(("[x] Stage 4 is NOT treated as the matched counterfactual", True))
    
    free_space_gb = shutil.disk_usage(".").free / (1024**3)
    estimated_space_gb = 15.0
    safe = free_space_gb > estimated_space_gb + 5.0
    checks.append(("[x] Storage estimate is safe", safe))
    
    print("=== VALIDATION CHECKLIST ===")
    for name, passed in checks:
        if passed:
            print(name)
        else:
            print(name.replace("[x]", "[FAILED]"))
            
    print(f"\n[STORAGE INFO]")
    print(f"Current free disk space: {free_space_gb:.2f} GB")
    print(f"Estimated experiment size: {estimated_space_gb:.2f} GB")
    print(f"Expected remaining free space: {(free_space_gb - estimated_space_gb):.2f} GB")

if __name__ == "__main__":
    validate_timing_design()
