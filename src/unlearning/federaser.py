import torch
import math
from typing import Dict, List, Tuple

def compute_l2_norm_of_floats(delta: Dict[str, torch.Tensor]) -> float:
    """
    Computes the global L2 norm of the flattened vector of all floating-point tensors
    in the state dictionary delta. This excludes BatchNorm integer buffers like num_batches_tracked.
    """
    squared_sum = 0.0
    for key, tensor in delta.items():
        if tensor.is_floating_point() and "running_" not in key and "num_batches_tracked" not in key:
            squared_sum += torch.sum(tensor ** 2).item()
            
    return math.sqrt(squared_sum)

def calibrate_update(historical_delta: Dict[str, torch.Tensor], 
                     calibration_delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Implements the FedEraser update calibration mathematics:
    tilde U = ||U||_2 * (hat U / ||hat U||_2)
    
    This is applied globally across all floating-point parameters (weights/biases).
    BatchNorm buffers (running_mean, running_var, num_batches_tracked) bypass magnitude
    scaling to prevent statistical corruption, taking their value directly from the calibration delta.
    
    Args:
        historical_delta: U (stored historical update)
        calibration_delta: hat U (new calibration update)
    """
    hist_norm = compute_l2_norm_of_floats(historical_delta)
    cali_norm = compute_l2_norm_of_floats(calibration_delta)
    
    calibrated_delta = {}
    
    # Zero-norm safeguard
    scale_factor = 1.0
    if cali_norm > 1e-8:
        scale_factor = hist_norm / cali_norm
        
    for key in historical_delta.keys():
        tensor = calibration_delta[key]
        
        # Apply scaling only to floating point parameters (weights/biases)
        if tensor.is_floating_point() and "running_" not in key and "num_batches_tracked" not in key:
            calibrated_delta[key] = tensor * scale_factor
        else:
            # BatchNorm / integer buffer bypass adaptation
            calibrated_delta[key] = tensor.clone()
            
    # Assert that the global norm of each calibrated update equals the historical update norm
    if cali_norm > 1e-8:
        calibrated_norm = compute_l2_norm_of_floats(calibrated_delta)
        assert math.isclose(calibrated_norm, hist_norm, rel_tol=1e-4, abs_tol=1e-6), f"Norm mismatch: {calibrated_norm} != {hist_norm}"
            
    return calibrated_delta
