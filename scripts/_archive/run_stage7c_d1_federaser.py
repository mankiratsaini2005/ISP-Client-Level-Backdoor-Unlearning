import os
import subprocess

attacked_run_dir = "results/stage7c_d1_attacked_20260911_230919"
unlearn_dir = "results/stage7c_d1/unlearning"
mcf_path = "results/stage7c_d1_mcf_20260911_230957/checkpoints/global_round_100.pt"

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

print("Stage 7C D1 FedEraser complete.")
