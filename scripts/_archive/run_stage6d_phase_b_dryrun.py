import os
import subprocess
import time

def run_cmd(cmd):
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

base_dir = "results/stage_6d_temporal"
early_source = os.path.join(base_dir, "early/attack")
early_model_path = os.path.join(early_source, "checkpoints/global_round_100.pt")
ref_model_path = os.path.join(base_dir, "matched_counterfactual/checkpoints/global_round_100.pt")

timestamp = time.strftime("%Y%m%d_%H%M%S")
smoke_dir = f"results/stage6d_phase_b_smoke_{timestamp}"
os.makedirs(smoke_dir, exist_ok=True)

subtraction_dir = os.path.join(smoke_dir, "subtraction")
kd_dir = os.path.join(smoke_dir, "kd")
federaser_dir = os.path.join(smoke_dir, "federaser")

# 1. Subtraction
run_cmd([
    "python", "scripts/unlearn_stage5a.py",
    "--source_dir", early_source,
    "--output_dir", subtraction_dir,
    "--target_client", "0"
])

subtraction_model_path = os.path.join(subtraction_dir, "checkpoints/subtraction_model.pt")

# 2. KD
run_cmd([
    "python", "scripts/unlearn_stage5b.py",
    "--teacher_path", early_model_path,
    "--student_path", subtraction_model_path,
    "--reference_path", ref_model_path,
    "--out_dir", kd_dir
])

# 3. FedEraser (Smoke Test)
run_cmd([
    "python", "scripts/run_federaser.py",
    "--source_dir", early_source,
    "--reference_path", ref_model_path,
    "--out_dir", federaser_dir,
    "--smoke-test"
])

print("\n=== DRY RUN COMPLETED ===")
