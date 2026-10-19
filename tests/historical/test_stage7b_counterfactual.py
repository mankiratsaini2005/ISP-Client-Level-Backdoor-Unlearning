import sys
import os
import yaml
import torch

# Mock the federated run by running it for 1 round, disable attack, exclude client 0
sys.path.insert(0, os.path.abspath("."))
try:
    from scripts._archive.run_federated import run_federated
except ImportError:
    from scripts.run_federated import run_federated

def test_counterfactual_exclusion():
    # Run federated for 2 rounds
    run_federated(
        config_path="configs/federated_baseline.yaml",
        mode="dev",
        rounds_override=2,
        run_prefix_override="test_stage7b_cf_",
        skip_client_updates=True,
        save_final_only=True,
        partition_type="H0",
        exclude_client_id=0,
        disable_attack=True
    )
    
    # Check the latest run dir
    runs = [d for d in os.listdir("results") if d.startswith("stage7b_test_stage7b_cf_")]
    # Wait, run_prefix is formatted as "stage7a_" in the script?
    runs = [d for d in os.listdir("results") if "test_stage7b_cf_" in d]
    if not runs:
        print("Run directory not found!")
        sys.exit(1)
        
    latest_run = max(runs, key=lambda d: os.path.getctime(os.path.join("results", d)))
    run_dir = os.path.join("results", latest_run)
    
    import json
    with open(os.path.join(run_dir, "client_sampling.json"), "r") as f:
        sampling = json.load(f)
        
    for r, clients in sampling.items():
        if 0 in clients:
            print(f"FAILED: Client 0 found in round {r}!")
            sys.exit(1)
            
        print(f"Round {r} clients: {clients} (Count: {len(clients)})")
        if len(clients) == 10:
            print("Client count is 10, meaning Client 0 was not originally drawn, or bug.")
        elif len(clients) == 9:
            print("Client count is 9, meaning Client 0 was excluded post-sampling! Correct!")
        
    print("SUCCESS: Client 0 exclusion verified.")

if __name__ == "__main__":
    test_counterfactual_exclusion()
