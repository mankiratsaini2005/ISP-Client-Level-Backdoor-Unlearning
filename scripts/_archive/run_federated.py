import os
import time
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import pandas as pd
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.data.cifar10 import get_cifar10_dataloaders
from src.models.small_cnn import SmallCNN
from src.training.train_utils import train_epoch, evaluate
from src.data.fl_partition import create_iid_partitions, save_partition_metadata
try:
    from scripts.stage7a_generate_partitions import create_controlled_dirichlet_partitions
except:
    pass
from src.training.fedavg import compute_model_delta, aggregate_fedavg, get_communication_bytes

try:
    from src.data.backdoor import PoisonedDataset, create_asr_dataloader
except ImportError:
    pass

def run_federated(config_path: str, mode: str = "dev", scale_factor_override: float = None, rounds_override: int = None, run_prefix_override: str = None, skip_client_updates: bool = False, save_final_only: bool = False, partition_type: str = "H0", exclude_client_id: int = None, disable_attack: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if exclude_client_id is not None:
        if "drop_clients_post_sampling" not in config["federated"]:
            config["federated"]["drop_clients_post_sampling"] = []
        config["federated"]["drop_clients_post_sampling"].append(exclude_client_id)
        
    if scale_factor_override is not None and "attack" in config:
        config["attack"]["scale_factor"] = scale_factor_override
        
    if rounds_override is not None:
        config["federated"][f"rounds_{mode}"] = rounds_override
        
    set_seed(config["seed"])
    
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_prefix_override is not None:
        run_prefix = run_prefix_override
    else:
        if "attack" in config:
            run_prefix = "attack_badnets_"
        elif "asr_eval" in config:
            run_prefix = "full_retraining_"
        else:
            run_prefix = "fl_iid_"
    run_dir = os.path.join(config["paths"]["results_dir"], f"{run_prefix}{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    updates_dir = os.path.join(run_dir, "client_updates")
    fig_dir = os.path.join(run_dir, "figures")
    
    for d in [run_dir, ckpt_dir, updates_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    # Determine device
    device_pref = config["training"].get("device_preference", "cpu") if "training" in config else config["federated"].get("device_preference", "cpu")
    if device_pref == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load raw datasets
    # We use get_cifar10_dataloaders but manually extract the underlying dataset to partition it.
    print("Loading CIFAR-10 data...")
    train_loader_full, test_loader = get_cifar10_dataloaders(
        data_dir=config["data"]["dataset_path"], 
        batch_size=config["data"]["batch_size"],
        use_fake=False
    )
    train_dataset_full = train_loader_full.dataset
    
    # Partition
    num_clients = config["federated"]["num_clients"]
    if partition_type == "H0":
        partitions, distributions = create_iid_partitions(train_dataset_full, num_clients, seed=config["seed"])
    elif partition_type == "H1":
        partitions, distributions = create_controlled_dirichlet_partitions(train_dataset_full, num_clients, alpha=0.5, seed=config["seed"])
    elif partition_type == "H2":
        partitions, distributions = create_controlled_dirichlet_partitions(train_dataset_full, num_clients, alpha=0.1, seed=config["seed"])
    elif partition_type in ["D1", "D2"]:
        idx_file = f"results/stage7c_partitions/{partition_type.lower()}_indices.json"
        with open(idx_file, "r") as f:
            idx_dict = json.load(f)
        partitions = [Subset(train_dataset_full, idx_dict[str(i)]) for i in range(num_clients)]
        
        dist_file = f"results/stage7c_partitions/{partition_type.lower()}_distribution.json"
        with open(dist_file, "r") as f:
            distributions = json.load(f)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    save_partition_metadata(distributions, os.path.join(run_dir, "client_partitions.json"))
    
    # Build client dataloaders
    batch_size = config["data"]["batch_size"]
    client_loaders = [DataLoader(p, batch_size=batch_size, shuffle=True, num_workers=0) for p in partitions]
    
    # --- Attack Setup ---
    is_attack_run = "attack" in config and not disable_attack
    if is_attack_run:
        atk_cfg = config["attack"]
        with open(os.path.join(run_dir, "attack_metadata.json"), "w") as f:
            json.dump({
                "target_client": atk_cfg.get("malicious_client", 0),
                "gamma": atk_cfg.get("scale_factor", 5.0) if scale_factor_override is None else scale_factor_override,
                "attack_start": atk_cfg.get("start_round", 30),
                "attack_end": atk_cfg.get("end_round", 100),
                "poison_ratio": atk_cfg.get("poison_ratio", 0.5),
                "trigger": atk_cfg.get("trigger_type", "badnets"),
                "target_class": atk_cfg.get("target_class", 0),
                "seed": config["seed"]
            }, f, indent=4)
        malicious_id = atk_cfg["malicious_client_id"]
        poison_ratio = atk_cfg["poison_ratio"]
        attack_start_round = atk_cfg.get("attack_start_round", 1)
        attack_end_round = atk_cfg.get("attack_end_round", 1000000)
        scale_factor = atk_cfg.get("scale_factor", 1.0)
        
        # Determine poison indices deterministically
        malicious_partition = partitions[malicious_id]
        num_malicious_samples = len(malicious_partition)
        num_poisoned = int(num_malicious_samples * poison_ratio)
        
        # Seeded selection for poison indices
        p_gen = torch.Generator().manual_seed(config["seed"] + 123)
        poison_indices = torch.randperm(num_malicious_samples, generator=p_gen)[:num_poisoned].tolist()
        
        # Create poisoned dataloader for the malicious client
        poisoned_dataset = PoisonedDataset(
            base_dataset=malicious_partition,
            poison_indices=poison_indices,
            target_class=atk_cfg["target_class"],
            trigger_size=atk_cfg["trigger_size"]
        )
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Create ASR dataloader
        asr_loader = create_asr_dataloader(test_loader, atk_cfg["target_class"], atk_cfg["trigger_size"], batch_size)
        
        attack_metadata = {
            "malicious_client_id": malicious_id,
            "poison_ratio": poison_ratio,
            "num_poisoned_samples": num_poisoned,
            "attack_start_round": attack_start_round,
            "attack_end_round": attack_end_round,
            "scale_factor": scale_factor
        }
        with open(os.path.join(run_dir, "attack_metadata.json"), "w") as f:
            json.dump(attack_metadata, f, indent=4)
    elif "asr_eval" in config:
        asr_cfg = config["asr_eval"]
        asr_loader = create_asr_dataloader(test_loader, asr_cfg["target_class"], asr_cfg["trigger_size"], batch_size)
        attack_start_round = float('inf')
        attack_end_round = -1
        malicious_id = -1
        scale_factor = 1.0
    else:
        attack_start_round = float('inf')
        attack_end_round = -1
        malicious_id = -1
        scale_factor = 1.0
        asr_loader = None
    
    # Init global model
    global_model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    rounds = config["federated"][f"rounds_{mode}"]
    clients_per_round = config["federated"]["clients_per_round"]
    
    metrics = []
    sampling_schedule = {}
    
    # Checkpoint initial model
    torch.save(global_model.state_dict(), os.path.join(ckpt_dir, "global_round_000.pt"))
    
    global_model_size_bytes = get_communication_bytes(global_model.state_dict())
    cum_comm_bytes = 0
    cum_time = 0.0
    best_acc = 0.0
    
    # Generator for sampling
    g = torch.Generator()
    g.manual_seed(config["seed"])
    
    print(f"Starting Federated {mode.upper()} run for {rounds} rounds...")
    for round_num in range(1, rounds + 1):
        round_start = time.time()
        
        # 1. Sample clients
        excluded_clients = config["federated"].get("exclude_clients", [])
        drop_post_sampling = config["federated"].get("drop_clients_post_sampling", [])
        
        if not excluded_clients:
            selected_clients = torch.randperm(num_clients, generator=g)[:clients_per_round].tolist()
        else:
            eligible_clients = [i for i in range(num_clients) if i not in excluded_clients]
            perm = torch.randperm(len(eligible_clients), generator=g).tolist()
            selected_clients = [eligible_clients[i] for i in perm[:clients_per_round]]
            
        if drop_post_sampling:
            selected_clients = [c for c in selected_clients if c not in drop_post_sampling]
            
        sampling_schedule[f"round_{round_num:03d}"] = selected_clients
        
        # 2. Extract global state
        global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
        
        # 3. Local training
        client_deltas = []
        sample_counts = []
        
        round_updates_dir = os.path.join(updates_dir, f"round_{round_num:03d}")
        os.makedirs(round_updates_dir, exist_ok=True)
        
        malicious_participated = False
        round_poisoned_samples = 0
        raw_update_l2_norm = 0.0
        scaled_update_l2_norm = 0.0
        
        for client_id in selected_clients:
            # Init fresh model with global state
            local_model = SmallCNN().to(device)
            local_model.load_state_dict(global_state)
            
            # Setup optimizer
            optimizer = torch.optim.SGD(
                local_model.parameters(), 
                lr=config["federated"]["learning_rate"],
                momentum=config["federated"]["momentum"],
                weight_decay=config["federated"]["weight_decay"]
            )
            
            # Train
            if is_attack_run and client_id == malicious_id and attack_start_round <= round_num <= attack_end_round:
                loader = poisoned_loader
                malicious_participated = True
                round_poisoned_samples = num_poisoned
            else:
                loader = client_loaders[client_id]
                
            for epoch in range(config["federated"]["local_epochs"]):
                train_epoch(local_model, loader, criterion, optimizer, device)
                
            # Compute delta and save
            local_state = {k: v.cpu() for k, v in local_model.state_dict().items()}
            raw_delta = compute_model_delta(global_state, local_state)
            
            num_samples = len(partitions[client_id])
            
            # Update scaling logic
            if is_attack_run and client_id == malicious_id and attack_start_round <= round_num <= attack_end_round:
                scaled_delta = {}
                # Calculate L2 norm of raw_delta
                raw_norm_sq = 0.0
                for k, v in raw_delta.items():
                    if v.is_floating_point():
                        raw_norm_sq += torch.sum(v ** 2).item()
                        scaled_delta[k] = v * scale_factor
                    else:
                        scaled_delta[k] = v # Don't scale integer trackers
                
                raw_update_l2_norm = (raw_norm_sq ** 0.5)
                
                scaled_norm_sq = 0.0
                for k, v in scaled_delta.items():
                    if v.is_floating_point():
                        scaled_norm_sq += torch.sum(v ** 2).item()
                scaled_update_l2_norm = (scaled_norm_sq ** 0.5)
                
                final_delta = scaled_delta
            else:
                final_delta = raw_delta
            
            client_deltas.append(final_delta)
            sample_counts.append(num_samples)
            
            # Save client delta
            save_dict = {
                "round": round_num,
                "client_id": client_id,
                "num_samples": num_samples,
                "delta": final_delta
            }
            if is_attack_run and client_id == malicious_id and attack_start_round <= round_num <= attack_end_round:
                save_dict["raw_delta"] = raw_delta
                save_dict["scale_factor"] = scale_factor
                
            if not skip_client_updates:
                torch.save(save_dict, os.path.join(round_updates_dir, f"client_{client_id:02d}.pt"))
            
        # 4. Aggregate
        new_global_state = aggregate_fedavg(global_state, client_deltas, sample_counts)
        global_model.load_state_dict(new_global_state)
        
        # 5. Evaluate
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        
        asr_acc = 0.0
        if asr_loader is not None:
            _, asr_acc = evaluate(global_model, asr_loader, criterion, device)
        
        round_time = time.time() - round_start
        cum_time += round_time
        
        round_dl_bytes = global_model_size_bytes * clients_per_round
        round_ul_bytes = global_model_size_bytes * clients_per_round
        cum_comm_bytes += (round_dl_bytes + round_ul_bytes)
        
        log_str = f"Round [{round_num:03d}/{rounds:03d}] - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        if asr_loader is not None:
            log_str += f", ASR: {asr_acc:.2f}%"
        log_str += f" - Time: {round_time:.1f}s"
        print(log_str)
        
        metrics_dict = {
            "round": round_num,
            "selected_clients": selected_clients,
            "num_participating": clients_per_round,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "round_time_s": round_time,
            "cumulative_time_s": cum_time,
            "download_bytes": round_dl_bytes,
            "upload_bytes": round_ul_bytes,
            "cumulative_communication_bytes": cum_comm_bytes
        }
        if asr_loader is not None:
            metrics_dict["asr"] = asr_acc
            
        if is_attack_run:
            metrics_dict["malicious_participated"] = malicious_participated
            metrics_dict["num_poisoned_samples"] = round_poisoned_samples
            metrics_dict["raw_update_l2_norm"] = raw_update_l2_norm
            metrics_dict["scaled_update_l2_norm"] = scaled_update_l2_norm
            
        metrics.append(metrics_dict)
        
        # Save checkpoints
        if not save_final_only or round_num == rounds:
            torch.save(new_global_state, os.path.join(ckpt_dir, f"global_round_{round_num:03d}.pt"))
            
            # Save RNG states for exact resumability
            import random
            import numpy as np
            rng_states = {
                'torch_rng': torch.get_rng_state(),
                'numpy_rng': np.random.get_state(),
                'python_rng': random.getstate(),
            }
            if torch.backends.mps.is_available():
                # MPS doesn't have a specific get_rng_state in older PyTorch, but we check if manual_seed works.
                pass
            torch.save(rng_states, os.path.join(ckpt_dir, f"rng_state_round_{round_num:03d}.pt"))
            
            # Write explicit round complete marker
            with open(os.path.join(run_dir, f"round_{round_num:03d}_complete.json"), "w") as f:
                json.dump({"round": round_num, "status": "complete", "timestamp": str(datetime.now())}, f)
                
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(new_global_state, os.path.join(ckpt_dir, "best_global.pt"))
            print(f"--> Saved new best model with acc {best_acc:.2f}%")
            
    # Save final artifacts
    with open(os.path.join(run_dir, "client_sampling.json"), "w") as f:
        json.dump(sampling_schedule, f, indent=4)
        
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    
    # Plot required figures
    try:
        import matplotlib.pyplot as plt
        
        # Test Acc vs Round
        plt.figure()
        plt.plot(df['round'], df['test_acc'], marker='o')
        plt.title('Global Test Accuracy vs Round')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "test_acc_vs_round.png"))
        plt.close()
        
        # Test Loss vs Round
        plt.figure()
        plt.plot(df['round'], df['test_loss'], marker='o')
        plt.title('Global Test Loss vs Round')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "test_loss_vs_round.png"))
        plt.close()
        
        # Time vs Round
        plt.figure()
        plt.plot(df['round'], df['round_time_s'], marker='o')
        plt.title('Wall-clock Time vs Round')
        plt.xlabel('Round')
        plt.ylabel('Time (s)')
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "time_vs_round.png"))
        plt.close()
        
        # Communication vs Round
        plt.figure()
        plt.plot(df['round'], df['cumulative_communication_bytes'] / (1024*1024), marker='o')
        plt.title('Cumulative Communication vs Round')
        plt.xlabel('Round')
        plt.ylabel('Cumulative Transfer (MB)')
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "comm_vs_round.png"))
        plt.close()
        
        if asr_loader is not None:
            # ASR vs Round
            plt.figure()
            plt.plot(df['round'], df['asr'], marker='x', color='red', label='ASR')
            plt.plot(df['round'], df['test_acc'], marker='o', color='blue', label='Clean Acc')
            if attack_start_round != float('inf'):
                plt.axvline(x=attack_start_round, color='black', linestyle='--', label='Attack Start')
            plt.title('Clean Accuracy and ASR vs Round')
            plt.xlabel('Round')
            plt.ylabel('Accuracy / ASR (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, "attack_metrics_vs_round.png"))
            plt.close()
        
    except ImportError:
        print("Matplotlib not found, skipping plot generation.")
        
    print(f"Federated run complete. Results saved to: {run_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/federated_baseline.yaml")
    parser.add_argument("--mode", type=str, choices=["dev", "main"], default="dev")
    parser.add_argument("--scale_factor", type=float, default=None, help="Override attack scale factor (gamma)")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    parser.add_argument("--run_prefix", type=str, default=None, help="Prefix for run directory")
    parser.add_argument("--skip_client_updates", action="store_true", help="Do not save client update tensors to save space")
    parser.add_argument("--save_final_only", action="store_true", help="Only save the round 0 and final global checkpoints to save space")
    parser.add_argument("--partition_type", type=str, default="H0", help="Partition type to use")
    parser.add_argument("--exclude_client_id", type=int, default=None, help="ID of client to explicitly drop post-sampling (for CF generation)")
    parser.add_argument("--disable_attack", action="store_true", help="Disable the attack completely (No-Attack Control)")
    args = parser.parse_args()
    
    run_federated(args.config, args.mode, args.scale_factor, args.rounds, args.run_prefix, 
                  args.skip_client_updates, args.save_final_only, args.partition_type, 
                  args.exclude_client_id, args.disable_attack)
