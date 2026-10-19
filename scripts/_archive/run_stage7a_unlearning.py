import subprocess
import os

runs = [
    # C1 (H0)
    {"name": "c1", "source": "results/stage7a_c1_attacked_20260910_233012", "mcf": "results/stage7a_c1_mcf_20260911_003109"},
    # C2 (H1)
    {"name": "c2", "source": "results/stage7a_c2_attacked_20260911_005839", "mcf": "results/stage7a_c2_mcf_20260911_015906"},
    # C3 (H2)
    {"name": "c3", "source": "results/stage7a_c3_attacked_20260911_022619", "mcf": "results/stage7a_c3_mcf_20260911_032646"},
    # C4 (H2+A1)
    {"name": "c4", "source": "results/stage7a_c4_attacked_20260911_035522", "mcf": "results/stage7a_c3_mcf_20260911_032646"} # Reuses C3 MCF
]

base_out = "results/stage7a_unlearning"
os.makedirs(base_out, exist_ok=True)

for run in runs:
    print(f"\n========================================================")
    print(f"RUNNING UNLEARNING FOR CONDITION: {run['name'].upper()}")
    print(f"========================================================")
    
    cond_out = os.path.join(base_out, run['name'])
    sub_dir = os.path.join(cond_out, "subtraction")
    kd_dir = os.path.join(cond_out, "kd")
    federaser_dir = os.path.join(cond_out, "federaser")
    
    os.makedirs(cond_out, exist_ok=True)
    
    source_model = os.path.join(run['source'], "checkpoints/best_global.pt")
    if not os.path.exists(source_model):
        source_model = os.path.join(run['source'], "checkpoints/global_round_100.pt")
        
    mcf_model = os.path.join(run['mcf'], "checkpoints/best_global.pt")
    if not os.path.exists(mcf_model):
        mcf_model = os.path.join(run['mcf'], "checkpoints/global_round_100.pt")
        
    # 1. Subtraction
    print("-> 1. Historical Subtraction")
    subprocess.run([
        "python", "scripts/unlearn_stage5a.py",
        "--source_dir", run['source'],
        "--output_dir", sub_dir,
        "--target_client", "0"
    ], check=True)
    
    sub_model = os.path.join(sub_dir, "checkpoints/subtraction_model.pt")
    
    # 2. Subtraction + KD
    print("-> 2. Knowledge Distillation")
    subprocess.run([
        "python", "scripts/unlearn_stage5b.py",
        "--teacher_path", source_model,
        "--student_path", sub_model,
        "--reference_path", mcf_model,
        "--out_dir", kd_dir
    ], check=True)
    
    # 3. FedEraser
    print("-> 3. FedEraser")
    subprocess.run([
        "python", "scripts/run_federaser.py",
        "--source_dir", run['source'],
        "--reference_path", mcf_model,
        "--out_dir", federaser_dir
    ], check=True)

print("\nALL STAGE 7A UNLEARNING RUNS COMPLETED SUCCESSFULLY.")
