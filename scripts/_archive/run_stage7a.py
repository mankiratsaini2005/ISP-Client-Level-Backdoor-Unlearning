import subprocess
import sys

runs = [
    # C1 (H0 + A0)
    {"prefix": "stage7a_c1_attacked_", "partition": "H0", "gamma": 5.0, "type": "attack"},
    {"prefix": "stage7a_c1_noattack_", "partition": "H0", "gamma": 5.0, "type": "control"},
    {"prefix": "stage7a_c1_mcf_", "partition": "H0", "gamma": 5.0, "type": "mcf"},
    
    # C2 (H1 + A0)
    {"prefix": "stage7a_c2_attacked_", "partition": "H1", "gamma": 5.0, "type": "attack"},
    {"prefix": "stage7a_c2_noattack_", "partition": "H1", "gamma": 5.0, "type": "control"},
    {"prefix": "stage7a_c2_mcf_", "partition": "H1", "gamma": 5.0, "type": "mcf"},
    
    # C3 (H2 + A0)
    {"prefix": "stage7a_c3_attacked_", "partition": "H2", "gamma": 5.0, "type": "attack"},
    {"prefix": "stage7a_c3_noattack_", "partition": "H2", "gamma": 5.0, "type": "control"},
    {"prefix": "stage7a_c3_mcf_", "partition": "H2", "gamma": 5.0, "type": "mcf"},
    
    # C4 (H2 + A1)
    {"prefix": "stage7a_c4_attacked_", "partition": "H2", "gamma": 10.0, "type": "attack"}
]

for run in runs:
    cmd = [
        "python", "-u", "scripts/run_federated.py",
        "--config", "configs/attack_badnets.yaml",
        "--mode", "main",
        "--run_prefix", run["prefix"],
        "--scale_factor", str(run["gamma"]),
        "--partition_type", run["partition"]
    ]
    
    if run["type"] == "control":
        cmd.extend(["--disable_attack", "--skip_client_updates", "--save_final_only"])
    elif run["type"] == "mcf":
        cmd.extend(["--disable_attack", "--skip_client_updates", "--save_final_only", "--exclude_client_id", "0"])
        
    print(f"\n========================================================", flush=True)
    print(f"LAUNCHING: {run['prefix']} ({run['partition']}, gamma={run['gamma']})", flush=True)
    print(f"========================================================", flush=True)
    
    subprocess.run(cmd, check=True)

print("\nALL STAGE 7A TRAINING RUNS COMPLETED SUCCESSFULLY.", flush=True)
