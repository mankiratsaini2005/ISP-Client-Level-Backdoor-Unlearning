import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_trajectory():
    dir_M_FE = "results/unlearning_federaser_20260909_224318"
    metrics_file = os.path.join(dir_M_FE, "metrics.csv")
    
    if not os.path.exists(metrics_file):
        print(f"Metrics not found at {metrics_file}")
        return
        
    df = pd.read_csv(metrics_file)
    
    os.makedirs("figures", exist_ok=True)
    
    # Accuracy and ASR vs Round
    plt.figure(figsize=(10, 5))
    plt.plot(df['round'], df['clean_acc'], label='Clean Accuracy', color='blue')
    plt.plot(df['round'], df['asr'], label='ASR', color='red')
    
    # Add horizontal lines for counterfactual (Stage 4)
    plt.axhline(y=74.48, color='blue', linestyle='--', alpha=0.5, label='M_R Clean Acc (74.48%)')
    plt.axhline(y=3.02, color='red', linestyle='--', alpha=0.5, label='M_R ASR (3.02%)')
    
    plt.xlabel('Reconstruction Round')
    plt.ylabel('Percentage (%)')
    plt.title('FedEraser Trajectory vs Counterfactual Reference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/federaser_trajectory.png')
    print("Saved figures/federaser_trajectory.png")
    
    # Cost-Fidelity Frontier Data Plot
    # We will just plot a static scatter plot using the data we have.
    methods = ['Historical Subtraction', 'Subtraction + KD', 'FedEraser', 'Full Retraining']
    costs = [4.85, 15.0, 1680.17, 1800.0]  # rough seconds
    fidelity = [10.00, 63.89, 87.40, 100.0]
    
    plt.figure(figsize=(8, 5))
    plt.scatter(costs, fidelity, color='purple', s=100)
    for i, txt in enumerate(methods):
        plt.annotate(txt, (costs[i], fidelity[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xscale('log')
    plt.xlabel('Unlearning Cost (Seconds) - Log Scale')
    plt.ylabel('Counterfactual Fidelity (Prediction Agreement %)')
    plt.title('Counterfactual Fidelity vs Unlearning Cost')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/cost_fidelity_frontier.png')
    print("Saved figures/cost_fidelity_frontier.png")

if __name__ == "__main__":
    plot_trajectory()
