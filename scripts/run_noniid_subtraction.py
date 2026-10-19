import torch
import os
import copy
import time
import json
from src.models.small_cnn import SmallCNN

def main():
    start_time = time.time()
    output_dir = "results_phase2/noniid_subtraction"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cpu") # Subtraction is just arithmetic, cpu is fine
    
    # Load partitions to calculate c0_weight
    partitions = torch.load("results_phase2/noniid_attack_logged/client_partitions.pt", map_location=device, weights_only=False)
    n_0 = len(partitions[0])
    N = sum(len(p) for p in partitions)
    c0_weight = n_0 / N
    
    final_model_path = "results_phase2/noniid_attack_logged/final_model.pt"
    m_attack_state = torch.load(final_model_path, map_location=device)
    m_sub_state = copy.deepcopy(m_attack_state)
    
    print(f"Starting historical subtraction for Client 0 across all 50 rounds... (Non-IID c0_weight = {c0_weight:.4f})")
    
    for t in range(1, 51):
        delta_path = f"results_phase2/noniid_attack_logged/trajectory/round_{t:03d}/delta_client_00.pt"
        if not os.path.exists(delta_path):
            raise FileNotFoundError(f"Missing delta: {delta_path}")
            
        delta = torch.load(delta_path, map_location=device)
        
        for key in m_sub_state.keys():
            if m_sub_state[key].is_floating_point():
                # Subtract (n_0 / N) * \Delta_0^(t)
                m_sub_state[key] = m_sub_state[key] - (c0_weight * delta[key])
            else:
                # num_batches_tracked: DO NOT subtract or modify
                pass
                
    out_path = os.path.join(output_dir, "final_model.pt")
    torch.save(m_sub_state, out_path)
    
    end_time = time.time()
    
    print("Subtraction complete.")
    with open(os.path.join(output_dir, "subtraction_runtime.json"), "w") as f:
        json.dump({"runtime_seconds": end_time - start_time, "c0_weight": c0_weight, "n_0": n_0, "N": N}, f, indent=4)
        
if __name__ == "__main__":
    main()
