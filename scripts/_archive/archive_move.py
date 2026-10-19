import os
import shutil
import json
import hashlib
from pathlib import Path

targets = [
    "results/stage_6d_temporal",
    "results/stage7a_c2_attacked_20260911_005839",
    "results/stage7a_c4_attacked_20260911_035522",
    "results/stage7c_d2_attacked_recovery_20260911_191151_20260911_191154",
    "results/stage7c_d1_attacked_20260911_182144"
]

def get_dir_stats(path):
    p = Path(path)
    if not p.exists(): return None
    size = 0
    files = []
    num_pt = 0
    has_updates = (p / "client_updates").exists()
    has_checkpoints = (p / "checkpoints").exists()
    for f in p.rglob("*"):
        if f.is_file():
            s = f.stat().st_size
            size += s
            files.append(str(f))
            if f.suffix == ".pt":
                num_pt += 1
    return {
        "size": size,
        "num_files": len(files),
        "num_pt": num_pt,
        "has_updates": has_updates,
        "has_checkpoints": has_checkpoints,
        "sample_files": files[:5] # keep a small sample
    }

pre_move = {}
for t in targets:
    st = get_dir_stats(t)
    if st:
        pre_move[t] = st

with open("results/archive_staging_pre_move_manifest.json", "w") as f:
    json.dump(pre_move, f, indent=4)
    
os.makedirs("results/archive_staging", exist_ok=True)

post_move = {}
for t in targets:
    if t in pre_move:
        dest = f"results/archive_staging/{Path(t).name}"
        os.rename(t, dest) # In-filesystem move
        st = get_dir_stats(dest)
        verified = (st["size"] == pre_move[t]["size"] and st["num_files"] == pre_move[t]["num_files"])
        post_move[t] = {
            "original_path": t,
            "new_path": dest,
            "size": st["size"],
            "num_files": st["num_files"],
            "classification": "ARCHIVE_STAGING",
            "reason": "Historical/Invalidated run moved for manual external upload",
            "move_verification_status": "PASS" if verified else "FAIL"
        }

with open("results/archive_staging/archive_staging_manifest.json", "w") as f:
    json.dump(post_move, f, indent=4)

