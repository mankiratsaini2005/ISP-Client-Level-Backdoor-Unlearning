import os
import yaml
import subprocess
import glob
import shutil

BASE_CONFIG_PATH = "configs/attack_badnets.yaml"
OUTPUT_DIR = "results/stage_6d_temporal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

conditions = {
    "early": {"attack_start_round": 1, "attack_end_round": 30},
    "middle": {"attack_start_round": 36, "attack_end_round": 56},
    "late": {"attack_start_round": 64, "attack_end_round": 100},
}

def get_latest_dir(prefix):
    dirs = glob.glob(f"results/{prefix}_*")
    return max(dirs, key=os.path.getmtime) if dirs else None

def run_condition(name, config_modifier=None, prefix="stage6d"):
    print(f"\n[{name.upper()}] Starting...")
    
    with open(BASE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    if config_modifier:
        config = config_modifier(config)
        
    config_path = os.path.join(OUTPUT_DIR, f"{name}_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # Create the isolated nested directory
    nested_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(nested_dir, exist_ok=True)
        
    cmd = ["python", "scripts/run_federated.py", "--config", config_path, "--mode", "main", "--run_prefix", f"{prefix}_{name}"]
    subprocess.run(cmd, check=True)
    
    latest_dir = get_latest_dir(f"{prefix}_{name}")
    if latest_dir:
        # Move everything from latest_dir to nested_dir / "attack" (or root if not attacked)
        target_sub = os.path.join(nested_dir, "attack") if name in ["early", "middle", "late"] else nested_dir
        if not os.path.exists(target_sub):
            shutil.move(latest_dir, target_sub)
        print(f"[{name.upper()}] Completed. Output saved to {target_sub}")

if __name__ == "__main__":
    # 1. Attacked Conditions
    for c_name, bounds in conditions.items():
        def mod(cfg, b=bounds):
            cfg["attack"]["attack_start_round"] = b["attack_start_round"]
            cfg["attack"]["attack_end_round"] = b["attack_end_round"]
            cfg["attack"]["scale_factor"] = 5.0
            return cfg
        run_condition(c_name, mod)

    # 2. No-Attack Control
    def no_attack_mod(cfg):
        del cfg["attack"]
        return cfg
    run_condition("no_attack", no_attack_mod)

    # 3. Matched Counterfactual
    def counterfactual_mod(cfg):
        if "attack" in cfg:
            del cfg["attack"]
        cfg["federated"]["drop_clients_post_sampling"] = [0]
        return cfg
    run_condition("matched_counterfactual", counterfactual_mod)
