import os
import json
import pandas as pd

conditions = ['c1', 'c2', 'c3', 'c4']
methods = ['subtraction', 'kd', 'federaser']

print("=== STAGE 7A UNLEARNING RESULTS ===\n")
print("| Regime | Method | Clean Acc | ASR | CF Agreement | CF Gap | Runtime |")
print("|---|---|---:|---:|---:|---:|---:|")

results = []

for c in conditions:
    for m in methods:
        dir_path = os.path.join("results/stage7a_unlearning", c, m)
        log_path = os.path.join(dir_path, "metrics.json")
        
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                d = json.load(f)
                
            clean_acc = d.get('clean_test_acc', 0.0)
            asr = d.get('asr', 0.0)
            cf_agree = d.get('fidelity_to_reference', 0.0)
            if cf_agree == 0.0:
                cf_agree = d.get('fidelity', 0.0)
                
            runtime = d.get('total_time_s', d.get('runtime_s', 0.0))
            cf_gap = 100.0 - cf_agree
            
            # Format
            c_upper = c.upper()
            m_format = "Subtraction" if m == "subtraction" else ("Subtraction+KD" if m == "kd" else "FedEraser")
            
            print(f"| {c_upper} | {m_format} | {clean_acc:.2f}% | {asr:.2f}% | {cf_agree:.2f}% | {cf_gap:.2f}% | {runtime:.1f}s |")
            results.append({
                "Regime": c_upper,
                "Method": m_format,
                "Clean Acc": f"{clean_acc:.2f}%",
                "ASR": f"{asr:.2f}%",
                "CF Agreement": f"{cf_agree:.2f}%",
                "CF Gap": f"{cf_gap:.2f}%",
                "Runtime": f"{runtime:.1f}s"
            })
        else:
            print(f"| {c.upper()} | {m} | MISSING |")

