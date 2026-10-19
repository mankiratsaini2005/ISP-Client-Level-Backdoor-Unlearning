import subprocess
import os
import time

os.makedirs("results", exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_prefix = f"stage7c_d2_attacked_recovery_{timestamp}_"

print(f"Starting D2 Recovery Run: {run_prefix}")
cmd = [
    "python", "-u", "scripts/run_federated.py",
    "--config", "configs/attack_badnets.yaml",
    "--mode", "main",
    "--run_prefix", run_prefix,
    "--partition_type", "D2",
    "--scale_factor", "5.0"
]

log_file = f"results/{run_prefix.strip('_')}.log"
with open(log_file, "w") as f:
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
print(f"Finished {run_prefix}. Log at {log_file}")
