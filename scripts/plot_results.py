import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

# 1. results/results_table.csv (already printed, but I will dump it to a file)
experiments = [
    ("IID_Clean", "results/iid_clean/final_results.json"),
    ("IID_Attack", "results/iid_attack/final_results.json"),
    ("Non-IID_Clean", "results/noniid_clean/final_results.json"),
    ("Non-IID_Attack", "results/noniid_attack/final_results.json")
]

with open("results/results_table.csv", "w") as f:
    f.write("Experiment,Clean_Acc,Clean_Loss,Macro_F1,Triggered_Rate,Untriggered_Rate,Runtime_min\n")
    for name, path in experiments:
        with open(path, "r") as p:
            d = json.load(p)
            f.write(f"{name},{d['clean_accuracy']},{d['clean_loss']},{d['clean_macro_f1']},{d['triggered_rate']},{d['untriggered_airplane_rate']},{d['runtime_seconds']/60:.1f}\n")

# 2. Plots
plt.figure(figsize=(10, 6))
for name in ["iid_clean", "iid_attack", "noniid_clean", "noniid_attack"]:
    df = pd.read_csv(f"results/{name}/metrics_per_round.csv")
    plt.plot(df['Round'], df['Clean_Acc'], label=name, marker='o' if 'attack' in name else '', markersize=4)
plt.title("Clean Accuracy vs Round")
plt.xlabel("Round")
plt.ylabel("Clean Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("results/clean_acc_vs_round.png")
plt.close()

plt.figure(figsize=(10, 6))
for name in ["iid_clean", "iid_attack", "noniid_clean", "noniid_attack"]:
    df = pd.read_csv(f"results/{name}/metrics_per_round.csv")
    plt.plot(df['Round'], df['Triggered_Target_Rate'], label=name, marker='o' if 'attack' in name else '', markersize=4)
plt.axvline(x=30, color='r', linestyle='--', label='Attack Start')
plt.title("Triggered Target Rate (ASR) vs Round")
plt.xlabel("Round")
plt.ylabel("Triggered Target Rate (%)")
plt.legend()
plt.grid(True)
plt.savefig("results/triggered_rate_vs_round.png")
plt.close()

# 3. Heatmaps
def plot_heatmap(hist_path, out_path, title):
    with open(hist_path, "r") as f:
        hist = json.load(f)
        
    data = np.zeros((10, 10))
    for i in range(10):
        c = hist[f"client_{i}"]["class_distribution"]
        for j in range(10):
            data[i, j] = c.get(str(j), 0)
            
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap="Blues", aspect="auto")
    plt.colorbar(label="Number of samples")
    plt.title(title)
    plt.xlabel("Class ID")
    plt.ylabel("Client ID")
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(j, i, int(data[i, j]), ha="center", va="center", color="black" if data[i, j] < data.max()/2 else "white", fontsize=8)
            
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

plot_heatmap("results/iid_clean/client_class_histograms.json", "results/iid_heatmap.png", "IID Client x Class Distribution")
plot_heatmap("results/noniid_clean/client_class_histograms.json", "results/noniid_heatmap.png", "Non-IID Client x Class Distribution")

print("All plots generated.")
