import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def calculate_displacement(m1_path, m2_path):
    m1 = torch.load(m1_path, weights_only=False)
    m2 = torch.load(m2_path, weights_only=False)
    
    norm_sq = 0.0
    for k in m1:
        if m1[k].is_floating_point():
            diff = m1[k] - m2[k].to(m1[k].device)
            norm_sq += torch.sum(diff ** 2).item()
    return norm_sq ** 0.5

def plot_metrics(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plot Metrics (Clean Acc and ASR)
    stages = ["Stage 3B\n(Backdoored)", "Stage 5A\n(Naive Subtraction)", "Stage 4\n(Gold Standard)"]
    clean_accs = [74.23, 10.30, 74.78]
    asrs = [87.26, 0.00, 3.13]
    
    x = np.arange(len(stages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy (%)', color='skyblue')
    bars2 = ax.bar(x + width/2, asrs, width, label='Attack Success Rate (%)', color='salmon')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Unlearning Baselines Evaluation')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clean_accuracy_and_asr_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Parameter Displacement
    d_3B_5A = 15.8991
    # Calc Stage 4 vs Stage 5A displacement
    m5A_path = "results/unlearning_5a_subtraction_20260909_131440/checkpoints/subtraction_model.pt"
    m4_path = "results/full_retraining_final_20260909_024627/checkpoints/global_round_100.pt"
    d_4_5A = calculate_displacement(m5A_path, m4_path)
    
    # Also calc Stage 3B vs Stage 4 for context
    m3B_path = "results/attack_scaling_final_gamma5_20260909_015903/checkpoints/global_round_100.pt"
    d_3B_4 = calculate_displacement(m3B_path, m4_path)
    
    comparisons = ["|| M_5A - M_3B ||_2", "|| M_5A - M_4 ||_2", "|| M_3B - M_4 ||_2"]
    dists = [d_3B_5A, d_4_5A, d_3B_4]
    
    plt.figure(figsize=(8, 6))
    plt.bar(comparisons, dists, color=['orange', 'mediumpurple', 'gray'])
    plt.ylabel('L2 Distance')
    plt.title('Parameter Displacement (L2 Norm)')
    
    for i, v in enumerate(dists):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "parameter_change.png"), dpi=300)
    plt.close()
    
    print("Plots saved to", out_dir)

if __name__ == "__main__":
    plot_metrics("results/unlearning_5a_subtraction_20260909_131440/figures")
