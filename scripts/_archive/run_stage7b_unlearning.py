import subprocess
import os

def get_latest_dir(prefix):
    runs = [d for d in os.listdir("results") if d.startswith(prefix)]
    if not runs: return None
    latest = max(runs, key=lambda d: os.path.getctime(os.path.join("results", d)))
    return os.path.join("results", latest)

runs = [
    # C1 (H0)
    {"name": "c1", "source": "results/stage7a_c1_attacked_20260910_233012", "mcf": get_latest_dir("stage7b_c1_mcf_")},
    # C2 (H1)
    {"name": "c2", "source": "results/stage7a_c2_attacked_20260911_005839", "mcf": get_latest_dir("stage7b_c2_mcf_")},
    # C3 (H2)
    {"name": "c3", "source": "results/stage7a_c3_attacked_20260911_022619", "mcf": get_latest_dir("stage7b_c3_mcf_")},
    # C4 (H2+A1)
    {"name": "c4", "source": "results/stage7a_c4_attacked_20260911_035522", "mcf": get_latest_dir("stage7b_c3_mcf_")}
]

base_out = "results/stage7b_unlearning"
os.makedirs(base_out, exist_ok=True)

for run in runs:
    if run['mcf'] is None:
        print(f"Missing MCF for {run['name']}")
        continue
        
    print(f"\n========================================================")
    print(f"RUNNING UNLEARNING FOR CONDITION: {run['name'].upper()}")
    print(f"========================================================")
    
    cond_out = os.path.join(base_out, run['name'])
    sub_dir = os.path.join(cond_out, "subtraction")
    kd_dir = os.path.join(cond_out, "kd")
    federaser_dir = os.path.join(cond_out, "federaser")
    
    os.makedirs(cond_out, exist_ok=True)
    
    # Strictly use global_round_100.pt
    source_model = os.path.join(run['source'], "checkpoints/global_round_100.pt")
    mcf_model = os.path.join(run['mcf'], "checkpoints/global_round_100.pt")
        
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # 1. Subtraction
    print("-> 1. Historical Subtraction")
    subprocess.run([
        "python", "scripts/unlearn_stage5a.py",
        "--source_dir", run['source'],
        "--output_dir", sub_dir,
        "--target_client", "0"
    ], check=True, env=env)
    
    sub_model = os.path.join(sub_dir, "checkpoints/subtraction_model.pt")
    
    # 2. Subtraction + KD
    print("-> 2. Knowledge Distillation")
    subprocess.run([
        "python", "scripts/unlearn_stage5b.py",
        "--teacher_path", source_model,
        "--student_path", sub_model,
        "--reference_path", mcf_model,
        "--out_dir", kd_dir
    ], check=True, env=env)
    
    # 3. FedEraser
    print("-> 3. FedEraser")
    subprocess.run([
        "python", "scripts/run_federaser.py",
        "--source_dir", run['source'],
        "--reference_path", mcf_model,
        "--out_dir", federaser_dir
    ], check=True, env=env)

print("\nALL STAGE 7B UNLEARNING RUNS COMPLETED SUCCESSFULLY.")
