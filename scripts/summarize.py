import json

experiments = [
    ("IID + Clean", "results/iid_clean/final_results.json"),
    ("IID + Attack", "results/iid_attack/final_results.json"),
    ("Non-IID + Clean", "results/noniid_clean/final_results.json"),
    ("Non-IID + Attack", "results/noniid_attack/final_results.json")
]

print("| Experiment | Clean Acc (%) | Clean Loss | Macro F1 | TTR/ASR (%) | Untriggered Target Rate (%) | Runtime (min) |")
print("|---|---|---|---|---|---|---|")

for name, path in experiments:
    with open(path, "r") as f:
        data = json.load(f)
    
    acc = f"{data['clean_accuracy']:.2f}"
    loss = f"{data['clean_loss']:.4f}"
    f1 = f"{data['clean_macro_f1']:.4f}"
    ttr = f"{data['triggered_rate']:.2f}"
    utr = f"{data['untriggered_airplane_rate']:.2f}"
    runtime = f"{data['runtime_seconds']/60:.1f}"
    
    print(f"| {name} | {acc} | {loss} | {f1} | {ttr} | {utr} | {runtime} |")
