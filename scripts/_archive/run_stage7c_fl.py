import subprocess
import os

os.makedirs("results", exist_ok=True)

tasks = [
    # Attacked
    ("D1", "stage7c_d1_attacked_", False),
    ("D2", "stage7c_d2_attacked_", False),
    # MCF
    ("D1", "stage7c_d1_mcf_", True),
    ("D2", "stage7c_d2_mcf_", True)
]

for partition, prefix, is_mcf in tasks:
    print(f"Starting {prefix}...")
    cmd = [
        "python", "-u", "scripts/run_federated.py",
        "--config", "configs/attack_badnets.yaml" if not is_mcf else "configs/federated_baseline.yaml",
        "--mode", "main",
        "--run_prefix", prefix,
        "--save_final_only",
        "--partition_type", partition
    ]
    if is_mcf:
        cmd.extend(["--skip_client_updates", "--exclude_client_id", "0", "--disable_attack"])
        
    log_file = f"results/{prefix.strip('_')}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Finished {prefix}.")

