import torch
import math
import pytest
from src.unlearning.federaser import compute_l2_norm_of_floats, calibrate_update

class TestFedEraserMath:
    def test_compute_l2_norm_of_floats(self):
        delta = {
            "conv.weight": torch.tensor([1.0, 2.0, 3.0]), # norm = sqrt(14)
            "conv.bias": torch.tensor([-1.0, 2.0]),       # norm = sqrt(5)
            # Total float sum_sq = 19 -> norm = sqrt(19) ~ 4.3589
            "bn.running_mean": torch.tensor([10.0, 20.0]), # Should be ignored!
            "bn.num_batches_tracked": torch.tensor(5, dtype=torch.long) # Should be ignored!
        }
        
        # Manually enable requires_grad to simulate standard weights/biases if needed,
        # but our function checks names.
        
        norm = compute_l2_norm_of_floats(delta)
        assert math.isclose(norm, math.sqrt(19), rel_tol=1e-5), f"Expected {math.sqrt(19)}, got {norm}"

    def test_calibrate_update(self):
        # Historical update U
        U = {
            "weight": torch.tensor([2.0, 0.0, 0.0]), # norm = 2.0
            "bn.running_mean": torch.tensor([10.0]), # Will be ignored for scaling
            "bn.num_batches_tracked": torch.tensor([1])
        }
        
        # New calibration update \hat{U}
        hat_U = {
            "weight": torch.tensor([0.0, 3.0, 4.0]), # norm = 5.0
            "bn.running_mean": torch.tensor([15.0]),
            "bn.num_batches_tracked": torch.tensor([2])
        }
        
        # Expected: scale_factor = ||U|| / ||hat_U|| = 2.0 / 5.0 = 0.4
        # Calibrated weight = [0.0, 3.0, 4.0] * 0.4 = [0.0, 1.2, 1.6]
        # Calibrated bn.running_mean = [15.0] (untouched from hat_U)
        
        tilde_U = calibrate_update(U, hat_U)
        
        assert torch.allclose(tilde_U["weight"], torch.tensor([0.0, 1.2, 1.6]))
        assert torch.allclose(tilde_U["bn.running_mean"], torch.tensor([15.0]))
        assert torch.equal(tilde_U["bn.num_batches_tracked"], torch.tensor([2]))

    def test_zero_norm_handling(self):
        U = {"weight": torch.tensor([2.0, 0.0])}
        # Zero calibration update
        hat_U = {"weight": torch.tensor([0.0, 0.0])}
        
        tilde_U = calibrate_update(U, hat_U)
        # Should return hat_U without NaN
        assert torch.allclose(tilde_U["weight"], torch.tensor([0.0, 0.0]))
        assert not torch.isnan(tilde_U["weight"]).any()
