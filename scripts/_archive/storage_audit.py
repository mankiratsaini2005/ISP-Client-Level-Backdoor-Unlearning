import os
import json
import csv
from pathlib import Path

results_dir = Path("results")

def get_dir_size(path="."):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    return total

def format_size(size_bytes):
    return f"{size_bytes / (1024**2):.1f} MB"

audit_data = []

archive_candidates = {
    "keep_local": [],
    "archive_external": [],
    "potential_duplicates": [],
    "regenerable": [],
    "do_not_touch": []
}

for item in os.listdir(results_dir):
    item_path = results_dir / item
    if not item_path.is_dir():
        continue
        
    size = get_dir_size(item_path)
    if size < 1024 * 1024: # skip tiny dirs < 1MB
        continue
        
    num_files = sum(1 for _ in item_path.rglob("*"))
    num_checkpoints = sum(1 for _ in item_path.rglob("*.pt"))
    has_client_updates = (item_path / "client_updates").exists()
    has_global = any(item_path.rglob("global_round_*.pt"))
    has_rng = any(item_path.rglob("rng_state_round_*.pt"))
    
    # Classification Logic
    name = item_path.name
    canonical = False
    active = False
    historical = False
    invalidated = False
    incomplete = False
    regenerable = False
    
    if "stage7c_d2_attacked_recovery_20260911_1944" in name:
        active = True
        cls = "CURRENT ACTIVE"
        archive_candidates["keep_local"].append(str(item_path))
    elif "stage7c_d2_attacked_recovery_20260911_1911" in name:
        invalidated = True
        cls = "INVALIDATED"
        archive_candidates["archive_external"].append(str(item_path))
    elif "stage7c_d2_attacked_20260911_1852" in name or "stage7c_d1_attacked_20260911_1821" in name:
        incomplete = True
        invalidated = True
        cls = "INCOMPLETE"
        archive_candidates["archive_external"].append(str(item_path))
    elif "stage_6d_temporal" in name:
        canonical = True
        historical = True
        cls = "CANONICAL HISTORICAL"
        archive_candidates["archive_external"].append(str(item_path)) # Large historical, can archive to external SSD
    elif "stage7a_" in name:
        # Note: Stage 7A outputs are technically invalidated by Stage 7B audit, BUT we use their attacked models!
        # Wait, Stage 7B re-evaluates Stage 7A models. Stage 7A attacked trajectories are canonical!
        if "attacked" in name:
            canonical = True
            historical = True
            cls = "CANONICAL"
            archive_candidates["do_not_touch"].append(str(item_path))
        else:
            invalidated = True
            cls = "INVALIDATED"
            archive_candidates["archive_external"].append(str(item_path))
    elif "stage7b" in name:
        canonical = True
        cls = "CANONICAL"
        archive_candidates["keep_local"].append(str(item_path))
    elif name in ["attack_scaling_final_gamma5_20260909_015903", "full_retraining_final_20260909_024627"]:
        canonical = True
        historical = True
        cls = "CANONICAL"
        archive_candidates["do_not_touch"].append(str(item_path))
    elif "attack_scaling_gamma" in name:
        historical = True
        cls = "HISTORICAL"
        archive_candidates["archive_external"].append(str(item_path))
    elif "fl_iid_" in name or "attack_badnets_" in name:
        historical = True
        cls = "HISTORICAL"
        archive_candidates["archive_external"].append(str(item_path))
    else:
        cls = "UNCATEGORIZED"
        archive_candidates["archive_external"].append(str(item_path))
        
    audit_data.append({
        "path": str(item_path),
        "size_mb": size / (1024**2),
        "num_files": num_files,
        "num_checkpoints": num_checkpoints,
        "client_updates": has_client_updates,
        "global_checkpoints": has_global,
        "rng_states": has_rng,
        "classification": cls
    })

# Write CSV
with open(results_dir / "storage_audit_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=audit_data[0].keys())
    writer.writeheader()
    for row in audit_data:
        row["size_mb"] = f"{row['size_mb']:.1f}"
        writer.writerow(row)
        
with open(results_dir / "storage_archive_candidates.json", "w") as f:
    json.dump(archive_candidates, f, indent=4)
    
print("Script finished")
