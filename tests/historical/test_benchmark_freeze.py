import os
import json
import torch
import pytest

from src.models.small_cnn import SmallCNN

class TestBenchmarkFreeze:
    """
    Validates that the canonical benchmark artifacts remain frozen, 
    accessible, and unmodified for future unlearning methods.
    """
    
    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.stage_3b_dir = "results/attack_scaling_final_gamma5_20260909_015903"
        self.stage_4_dir = "results/full_retraining_final_20260909_024627"
        self.stage_5a_dir = "results/unlearning_5a_subtraction_20260909_131440"
        self.stage_5b_dir = "results/unlearning_5b_kd_20260909_183734"
        
        self.stage_3b_ckpt = os.path.join(self.stage_3b_dir, "checkpoints/global_round_100.pt")
        self.stage_4_ckpt = os.path.join(self.stage_4_dir, "checkpoints/global_round_100.pt")
        self.stage_5a_ckpt = os.path.join(self.stage_5a_dir, "checkpoints/subtraction_model.pt")
        self.stage_5b_ckpt = os.path.join(self.stage_5b_dir, "checkpoints/final_unlearned.pt")
        
        self.proxy_indices_file = os.path.join(self.stage_5b_dir, "proxy_indices.json")
        self.eval_indices_file = os.path.join(self.stage_5b_dir, "eval_indices.json")

    def test_canonical_directories_exist(self):
        assert os.path.isdir(self.stage_3b_dir), "Stage 3B directory missing"
        assert os.path.isdir(self.stage_4_dir), "Stage 4 directory missing"
        assert os.path.isdir(self.stage_5a_dir), "Stage 5A directory missing"
        assert os.path.isdir(self.stage_5b_dir), "Stage 5B directory missing"

    def test_canonical_checkpoints_load(self):
        # Verify that models can be loaded seamlessly into the architecture
        model = SmallCNN()
        
        for ckpt in [self.stage_3b_ckpt, self.stage_4_ckpt, self.stage_5a_ckpt, self.stage_5b_ckpt]:
            assert os.path.isfile(ckpt), f"Checkpoint missing: {ckpt}"
            try:
                state_dict = torch.load(ckpt, weights_only=False)
                model.load_state_dict(state_dict)
            except Exception as e:
                pytest.fail(f"Failed to load checkpoint {ckpt}: {str(e)}")

    def test_evaluation_splits_frozen(self):
        assert os.path.isfile(self.proxy_indices_file), "Proxy indices missing"
        assert os.path.isfile(self.eval_indices_file), "Eval indices missing"
        
        with open(self.proxy_indices_file, "r") as f:
            proxy_indices = json.load(f)
        with open(self.eval_indices_file, "r") as f:
            eval_indices = json.load(f)
            
        assert len(proxy_indices) == 2500, f"Expected 2500 proxy indices, got {len(proxy_indices)}"
        assert len(eval_indices) == 7500, f"Expected 7500 eval indices, got {len(eval_indices)}"
        
        # Verify they are disjoint
        assert len(set(proxy_indices).intersection(set(eval_indices))) == 0, "Proxy and Eval sets overlap!"

    def test_target_client_and_participation_history_frozen(self):
        sampling_file = os.path.join(self.stage_3b_dir, "client_sampling.json")
        assert os.path.isfile(sampling_file), "Client sampling history missing"
        
        with open(sampling_file, "r") as f:
            sampling = json.load(f)
            
        assert len(sampling) == 100, "Expected exactly 100 federated rounds"
        
        # Target client is 0. Verify participation count.
        client_0_participations = sum(1 for clients in sampling.values() if 0 in clients)
        assert client_0_participations == 50, f"Client 0 should have exactly 50 participations, found {client_0_participations}"

    def test_data_partition_frozen(self):
        partition_file = os.path.join(self.stage_3b_dir, "client_partitions.json")
        assert os.path.isfile(partition_file), "Client partition history missing"
        
        with open(partition_file, "r") as f:
            partitions = json.load(f)
            
        assert len(partitions) == 20, "Expected exactly 20 client partitions"
        
        # Verify 2500 samples per client
        for client_id, client_info in partitions.items():
            assert client_info["num_samples"] == 2500, f"Client {client_id} has {client_info['num_samples']} samples instead of 2500"
