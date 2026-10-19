import subprocess
import os

def run_mcf(partition, prefix):
    print(f"Starting MCF for {partition}...")
    cmd = [
        "python", "-u", "scripts/run_federated.py",
        "--config", "configs/federated_baseline.yaml",
        "--mode", "main",
        "--run_prefix", f"stage7b_{prefix}_mcf_",
        "--save_final_only",
        "--skip_client_updates",
        "--partition_type", partition,
        "--exclude_client_id", "0",
        "--disable_attack"
    ]
    log_file = f"results/stage7b_{prefix}_mcf.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Finished MCF for {partition}.")

if __name__ == "__main__":
    import multiprocessing
    os.makedirs("results", exist_ok=True)
    tasks = [
        ("H0", "c1"),
        ("H1", "c2"),
        ("H2", "c3")
    ]
    # Run sequentially to avoid resource contention (MPS issues with multiprocessing)
    for p, pr in tasks:
        run_mcf(p, pr)
