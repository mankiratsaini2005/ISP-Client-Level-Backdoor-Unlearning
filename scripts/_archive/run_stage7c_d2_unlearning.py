import os
import subprocess

def run_cmd(cmd):
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

cond_source = "results/stage7c_d2_attacked_recovery_20260911_194422_20260911_194423"
ref_model_path = "results/stage7c_d2_mcf20260911_213425/checkpoints/global_round_100.pt"

full_dir = "results/stage7c_d2/unlearning"
os.makedirs(full_dir, exist_ok=True)

subtraction_dir = os.path.join(full_dir, "subtraction")
kd_dir = os.path.join(full_dir, "kd")
federaser_dir = os.path.join(full_dir, "federaser")

# 1. Subtraction
run_cmd([
    ".venv/bin/python", "scripts/unlearn_stage5a.py",
    "--source_dir", cond_source,
    "--output_dir", subtraction_dir,
    "--target_client", "0"
])

subtraction_model_path = os.path.join(subtraction_dir, "checkpoints/subtraction_model.pt")

# 2. KD
run_cmd([
    ".venv/bin/python", "scripts/unlearn_stage5b.py",
    "--teacher_path", os.path.join(cond_source, "checkpoints/global_round_100.pt"),
    "--student_path", subtraction_model_path,
    "--reference_path", ref_model_path,
    "--out_dir", kd_dir
])

# 3. FedEraser
run_cmd([
    ".venv/bin/python", "scripts/run_federaser.py",
    "--source_dir", cond_source,
    "--reference_path", ref_model_path,
    "--out_dir", federaser_dir
])

print("\n=== D2 UNLEARNING COMPLETED ===")
