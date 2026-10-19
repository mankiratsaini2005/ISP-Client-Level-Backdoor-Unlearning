import torch
import numpy as np
import random
from src.utils.seed import set_seed

def test_seed_reproducibility():
    set_seed(42)
    val1 = random.random()
    np_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()
    
    set_seed(42)
    val2 = random.random()
    np_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()
    
    assert val1 == val2
    assert np_val1 == np_val2
    assert torch_val1 == torch_val2
