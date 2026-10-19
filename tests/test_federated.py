import os
import torch
import torch.nn as nn
from torchvision.datasets import FakeData
from src.data.fl_partition import create_iid_partitions
from src.training.fedavg import compute_model_delta, aggregate_fedavg, get_communication_bytes

def test_partitioning():
    # 50,000 samples, 20 clients -> 2500 each
    dataset = FakeData(size=50000, image_size=(3, 32, 32), num_classes=10)
    partitions, distributions = create_iid_partitions(dataset, num_clients=20, seed=42)
    
    assert len(partitions) == 20
    
    all_indices = set()
    for i, p in enumerate(partitions):
        assert len(p) == 2500
        indices = set(p.indices)
        assert len(indices.intersection(all_indices)) == 0  # No overlap
        all_indices.update(indices)
        
    assert len(all_indices) == 50000 # All samples covered

def test_fedavg_arithmetic():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1, bias=False)
            self.register_buffer("counter", torch.tensor([0], dtype=torch.int64))

    global_model = DummyModel()
    global_model.linear.weight.data = torch.tensor([[1.0, 1.0]])
    global_model.counter.data = torch.tensor([1])
    
    global_state = global_model.state_dict()
    
    # Client 1: wt = [2.0, 2.0], count = [2], delta = [1.0, 1.0], [1]
    client1_delta = {
        "linear.weight": torch.tensor([[1.0, 1.0]]),
        "counter": torch.tensor([1], dtype=torch.int64)
    }
    
    # Client 2: wt = [4.0, 4.0], count = [4], delta = [3.0, 3.0], [3]
    client2_delta = {
        "linear.weight": torch.tensor([[3.0, 3.0]]),
        "counter": torch.tensor([3], dtype=torch.int64)
    }
    
    # Client 1 has 10 samples, Client 2 has 30 samples. Total = 40.
    # W_new = W_old + (10/40 * [1,1] + 30/40 * [3,3]) = 1 + (0.25 + 2.25) = 3.5
    # C_new = 1 + (0.25*1 + 0.75*3) = 1 + (0.25 + 2.25) = 3.5 -> round to 4
    
    new_global = aggregate_fedavg(global_state, [client1_delta, client2_delta], [10, 30])
    
    assert torch.allclose(new_global["linear.weight"], torch.tensor([[3.5, 3.5]]))
    assert new_global["counter"].item() == 4
    assert new_global["counter"].dtype == torch.int64

def test_compute_delta():
    global_state = {"w": torch.tensor([1.0])}
    local_state = {"w": torch.tensor([3.0])}
    
    delta = compute_model_delta(global_state, local_state)
    assert torch.allclose(delta["w"], torch.tensor([2.0]))
