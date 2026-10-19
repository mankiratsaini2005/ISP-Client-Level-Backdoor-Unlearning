import os
import subprocess

attacked_run_dir = "results/attack_badnets_20260912_182946"
unlearn_dir = "results/stage7c_d1_canonical/unlearning"
mcf_path = "results/stage7c_d1_mcf_20260911_230957/checkpoints/global_round_100.pt"

os.makedirs(unlearn_dir, exist_ok=True)

print("Running Historical Subtraction...")
subtraction_dir = os.path.join(unlearn_dir, "subtraction")
os.makedirs(subtraction_dir, exist_ok=True)
cmd1 = [
    "PYTHONPATH=.",
    ".venv/bin/python",
    "scripts/unlearn_stage5a.py",
    "--source_dir", attacked_run_dir,
    "--target_client", "0",
    "--output_dir", subtraction_dir
]
subprocess.run(" ".join(cmd1), shell=True, check=True)

print("Running KD...")
subtraction_model = os.path.join(subtraction_dir, "checkpoints", "subtraction_model.pt")
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
    "--source_dir", attacked_run_dir,
    "--reference_path", mcf_path,
    "--out_dir", federaser_dir
]
subprocess.run(" ".join(cmd3), shell=True, check=True)

print("Stage 7C D1 Canonical Unlearning complete.")
