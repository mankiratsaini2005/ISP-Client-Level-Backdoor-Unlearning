import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
import math

from src.training.fedavg import aggregate_fedavg, compute_model_delta
from src.unlearning.federaser import calibrate_update, compute_l2_norm_of_floats

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.bn = nn.BatchNorm1d(2)
        
    def forward(self, x):
        return self.bn(self.fc(x))

def run_toy_fl_test():
    torch.manual_seed(42)
    device = torch.device('cpu')
    
    # 5 clients, 1 forgotten (Client 0)
    num_clients = 5
    samples_per_client = 100
    
    # Mock data
    client_datasets = []
    for _ in range(num_clients):
        X = torch.randn(samples_per_client, 10)
        y = torch.randint(0, 2, (samples_per_client,))
        client_datasets.append(TensorDataset(X, y))
        
    # Mock historical artifacts
    global_model = TinyNet()
    global_state = global_model.state_dict()
    
    # Round 1: Client 0, 1, 2 participate
    # Round 2: Client 1, 3, 4 participate
    client_sampling = {
        1: [0, 1, 2],
        2: [1, 3, 4]
    }
    
    # We will just generate some random historical updates (deltas)
    historical_updates = {1: {}, 2: {}}
    for t in [1, 2]:
        for c in client_sampling[t]:
            delta = {}
            for k, v in global_state.items():
                if v.is_floating_point():
                    delta[k] = torch.randn_like(v) * 0.1
                else:
                    # integer buffers e.g. num_batches_tracked
                    delta[k] = torch.tensor(t, dtype=torch.long)
            historical_updates[t][c] = delta

    # Now, FedEraser Reconstruction!
    target_client = 0
    reconstructed_model = TinyNet()
    reconstructed_model.load_state_dict(global_state)
    
    for t in [1, 2]:
        sampled_clients = client_sampling[t]
        retained_clients = [c for c in sampled_clients if c != target_client]
        
        # Verify 1: Client 0 excluded
        assert target_client not in retained_clients
        
        client_calibrated_deltas = []
        client_sample_counts = []
        
        for c in retained_clients:
            U = historical_updates[t][c]
            
            # Local calibration (just mock one step)
            local_model = TinyNet()
            local_model.load_state_dict(reconstructed_model.state_dict())
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.1)
            loader = DataLoader(client_datasets[c], batch_size=32)
            
            local_model.train()
            for _ in range(2): # 2 epochs
                for X, y in loader:
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(local_model(X), y)
                    loss.backward()
                    optimizer.step()
            
            hat_U = compute_model_delta(reconstructed_model.state_dict(), local_model.state_dict())
            
            tilde_U = calibrate_update(U, hat_U)
            
            # Verify 7: Historical update calibration is correct
            norm_U = compute_l2_norm_of_floats(U)
            norm_tilde_U = compute_l2_norm_of_floats(tilde_U)
            assert math.isclose(norm_U, norm_tilde_U, rel_tol=1e-4)
            
            client_calibrated_deltas.append(tilde_U)
            client_sample_counts.append(samples_per_client)
            
        # Verify 5: Aggregation Weights
        total_samples = sum(client_sample_counts)
        weights = [n / total_samples for n in client_sample_counts]
        if t == 1:
            # retained = [1, 2], so 2 clients
            assert len(weights) == 2
            assert all(w == 0.5 for w in weights)
        else:
            # retained = [1, 3, 4], so 3 clients
            assert len(weights) == 3
            assert all(math.isclose(w, 1.0/3.0) for w in weights)
            
        assert math.isclose(sum(weights), 1.0)
        
        # Aggregate
        new_state = aggregate_fedavg(reconstructed_model.state_dict(), client_calibrated_deltas, client_sample_counts)
        reconstructed_model.load_state_dict(new_state)
        
    print("TOY FL TEST: PASSED")

if __name__ == "__main__":
    run_toy_fl_test()
