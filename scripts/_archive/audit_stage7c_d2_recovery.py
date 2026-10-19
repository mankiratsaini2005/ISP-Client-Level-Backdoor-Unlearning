import os
import json
import torch
import numpy as np

def get_latest_dir(prefix):
    runs = [d for d in os.listdir("results") if d.startswith(prefix) and os.path.isdir(os.path.join("results", d))]
    if not runs: return None
    latest = max(runs, key=lambda d: os.path.getctime(os.path.join("results", d)))
    return os.path.join("results", latest)

run_dir = get_latest_dir("stage7c_d2_attacked_recovery_")
if not run_dir:
    print("Could not find the recovery run directory.")
    exit(1)

print(f"Auditing Run Directory: {run_dir}")

# 1. Verify Checkpoints
ckpt_dir = os.path.join(run_dir, "checkpoints")
for i in range(101):
    ckpt = os.path.join(ckpt_dir, f"global_round_{i:03d}.pt")
    if not os.path.exists(ckpt):
        print(f"ERROR: Missing global_round_{i:03d}.pt")
        exit(1)
        
    rng_ckpt = os.path.join(ckpt_dir, f"rng_state_round_{i:03d}.pt")
    if not os.path.exists(rng_ckpt) and i > 0:
        print(f"ERROR: Missing rng_state_round_{i:03d}.pt")
        exit(1)

print("Checkpoints: All 101 global rounds (000-100) and RNG states are present.")

# 2. Verify Completion Markers
for i in range(1, 101):
    marker = os.path.join(run_dir, f"round_{i:03d}_complete.json")
    if not os.path.exists(marker):
        print(f"ERROR: Missing completion marker for round {i}")
        exit(1)

print("Completion markers: All 100 markers are present.")

# 3. Verify Client Updates
updates_dir = os.path.join(run_dir, "client_updates")
for i in range(1, 101): # Wait, rounds are 001 to 100? Or 000 to 099?
    # Actually run_federated creates client_updates/round_xxx.
    pass

# Actually run_federated saves client updates as round_{round_num:03d} where round_num is 1 to 100.
for i in range(1, 101):
    r_dir = os.path.join(updates_dir, f"round_{i:03d}")
    if not os.path.exists(r_dir):
        print(f"ERROR: Missing client_updates for round {i}")
        exit(1)
    pts = [f for f in os.listdir(r_dir) if f.endswith(".pt")]
    if len(pts) != 10:
        print(f"ERROR: Expected 10 client updates in round {i}, found {len(pts)}")
        exit(1)
        
print("Client updates: Complete for all 100 rounds (10 clients per round).")

# 4. Verify Partition Identity
with open(os.path.join(run_dir, "client_partitions.json"), "r") as f:
    part = json.load(f)
    
sizes = [part[f"client_{i}"]["num_samples"] for i in range(1, 20)]
if sizes[0] != 426 or sizes[1] != 12266: # sanity check sizes
    print("ERROR: Partition does not match D2 size specification.")
    exit(1)
print("Partition: D2 size imbalance verified.")

# 5. Attack Metadata
with open(os.path.join(run_dir, "attack_metadata.json"), "r") as f:
    atk = json.load(f)

if atk["scale_factor"] != 5.0 or atk["attack_start_round"] != 30 or atk["malicious_client_id"] != 0:
    print("ERROR: Attack metadata does not match A0 canonical configuration.")
    exit(1)
print("Attack configuration: Canonical A0 configuration verified.")

# 6. Verify NaN/Infinity in Round 100
m100 = torch.load(os.path.join(ckpt_dir, "global_round_100.pt"), map_location="cpu")
for k, v in m100.items():
    if torch.isnan(v.float()).any():
        print(f"ERROR: NaN found in {k}")
        exit(1)
    if torch.isinf(v.float()).any():
        print(f"ERROR: Infinity found in {k}")
        exit(1)

print("Round 100 Model: Finite, no NaNs.")

print("\n--- STAGE 7C D2 RECOVERY AUDIT SUCCESSFUL ---")

