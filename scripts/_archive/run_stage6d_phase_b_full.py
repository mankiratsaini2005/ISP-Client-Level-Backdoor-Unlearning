import os
import subprocess
import time

def run_cmd(cmd):
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

base_dir = "results/stage_6d_temporal"
ref_model_path = os.path.join(base_dir, "matched_counterfactual/checkpoints/global_round_100.pt")

timestamp = time.strftime("%Y%m%d_%H%M%S")
full_dir = f"results/stage6d_phase_b_full_{timestamp}"
os.makedirs(full_dir, exist_ok=True)

conditions = ["early", "middle", "late"]

for condition in conditions:
    print(f"\n{'='*50}\nStarting {condition.upper()} Condition\n{'='*50}")
    cond_source = os.path.join(base_dir, f"{condition}/attack")
    cond_model_path = os.path.join(cond_source, "checkpoints/global_round_100.pt")
    
    subtraction_dir = os.path.join(full_dir, condition, "subtraction")
    kd_dir = os.path.join(full_dir, condition, "kd")
    federaser_dir = os.path.join(full_dir, condition, "federaser")
    
    # 1. Subtraction
    run_cmd([
        "python", "scripts/unlearn_stage5a.py",
        "--source_dir", cond_source,
        "--output_dir", subtraction_dir,
        "--target_client", "0"
    ])
    
    subtraction_model_path = os.path.join(subtraction_dir, "checkpoints/subtraction_model.pt")
    
    # 2. KD
    run_cmd([
        "python", "scripts/unlearn_stage5b.py",
        "--teacher_path", cond_model_path,
        "--student_path", subtraction_model_path,
        "--reference_path", ref_model_path,
        "--out_dir", kd_dir
    ])
    
    # 3. FedEraser (Full 100 Rounds)
    run_cmd([
        "python", "scripts/run_federaser.py",
        "--source_dir", cond_source,
        "--reference_path", ref_model_path,
        "--out_dir", federaser_dir
    ])

print("\n=== FULL PHASE B-1 COMPLETED ===")
