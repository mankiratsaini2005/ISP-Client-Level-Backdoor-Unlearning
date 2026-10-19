import os
import subprocess
import shutil

attacked_run_dir = "results/stage7c_d1_attacked_20260911_230919"
unlearn_dir = "results/stage7c_d1/unlearning"
subtraction_dir = os.path.join(unlearn_dir, "subtraction")
subtraction_model = os.path.join(subtraction_dir, "checkpoints", "subtraction_model.pt")

print("Running KD...")
kd_dir = os.path.join(unlearn_dir, "kd")
os.makedirs(kd_dir, exist_ok=True)
cmd2 = [
    "PYTHONPATH=.",
    ".venv/bin/python",
    "scripts/unlearn_stage5b.py",
    "--teacher_path", os.path.join(attacked_run_dir, "checkpoints", "global_round_100.pt"),
    "--student_path", subtraction_model,
    "--reference_path", os.path.join(attacked_run_dir, "checkpoints", "global_round_100.pt"),
    "--out_dir", kd_dir
]
subprocess.run(" ".join(cmd2), shell=True, check=True)

print("Running FedEraser...")
federaser_dir = os.path.join(unlearn_dir, "federaser")
os.makedirs(federaser_dir, exist_ok=True)
cmd3 = [
    "PYTHONPATH=.",
    ".venv/bin/python",
    "scripts/run_federaser.py",
    "--attacked_dir", attacked_run_dir,
    "--output_dir", federaser_dir,
    "--target_client", "0"
]
subprocess.run(" ".join(cmd3), shell=True, check=True)

print("Stage 7C D1 Unlearning complete.")
