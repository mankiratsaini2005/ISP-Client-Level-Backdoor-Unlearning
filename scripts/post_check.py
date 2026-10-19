import json

with open("results/noniid_clean/client_class_histograms.json") as f:
    hist = json.load(f)
    
c0 = hist["client_0"]
total = c0["num_samples"]
dist = c0["class_distribution"]
airplane = dist.get("0", 0)
non_airplane = total - airplane
poisoned = int(total * 0.1)

print("=== C0 Non-IID Stats ===")
print(f"Total samples: {total}")
print(f"Class histogram: {dist}")
print(f"Non-Airplane samples: {non_airplane}")
print(f"Poisoned samples: {poisoned}")

with open("results/iid_attack/config.json") as f:
    config = json.load(f)
    
print("\n=== Attack Config ===")
print(f"poison ratio: {config['poison_ratio']}")
print(f"attack start round: {config['attack_start_round']}")
print(f"target class: {config['target_class']}")
print(f"trigger size: {config['trigger_size']}")
print(f"all-10-client participation: {config['num_clients'] == 10 and config['clients_per_round'] == 10}")

