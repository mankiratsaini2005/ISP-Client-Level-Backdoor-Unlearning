import torch
import copy
from typing import Dict, List, Tuple

def compute_model_delta(global_state: Dict[str, torch.Tensor], local_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Computes the delta (local_model - global_model) for all keys in the state_dict.
    """
    delta = {}
    for key in global_state.keys():
        if key in local_state:
            # We cast everything to float just to compute the exact delta if it's integer
            delta[key] = local_state[key].float() - global_state[key].float()
    return delta

def aggregate_fedavg(global_state: Dict[str, torch.Tensor], 
                     client_deltas: List[Dict[str, torch.Tensor]], 
                     sample_counts: List[int]) -> Dict[str, torch.Tensor]:
    """
    Performs standard weighted FedAvg aggregation using model deltas.
    
    w_{t+1} = w_t + \sum_k \frac{n_k}{N} \Delta_k
    """
    total_samples = sum(sample_counts)
    if total_samples == 0:
        return copy.deepcopy(global_state)
        
    weights = [n_k / total_samples for n_k in sample_counts]
    
    new_global_state = copy.deepcopy(global_state)
    
    for key in global_state.keys():
        # First, ensure all deltas have this key
        if not all(key in delta for delta in client_deltas):
            continue
            
        # Accumulate the weighted deltas
        accumulated_delta = torch.zeros_like(global_state[key], dtype=torch.float32)
        for delta, weight in zip(client_deltas, weights):
            accumulated_delta += delta[key].to(accumulated_delta.device) * weight
            
        # Add back to the global state
        new_val = global_state[key].float() + accumulated_delta
        
        # Handle type casting back (e.g. for batchnorm num_batches_tracked)
        if not global_state[key].is_floating_point():
            new_val = torch.round(new_val).to(global_state[key].dtype)
        else:
            new_val = new_val.to(global_state[key].dtype)
            
        new_global_state[key] = new_val
        
    return new_global_state

def get_communication_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Calculates the exact theoretical byte size of a state_dict for communication accounting.
    """
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes
