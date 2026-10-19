import os
import glob
import pandas as pd
import json

base_dir = "results/stage_6d_temporal"

runs = {
    "early": os.path.join(base_dir, "early", "attack"),
    "middle": os.path.join(base_dir, "middle", "attack"),
    "late": os.path.join(base_dir, "late", "attack"),
    "no_attack": os.path.join(base_dir, "no_attack"),
    "matched_counterfactual": os.path.join(base_dir, "matched_counterfactual")
}

def load_metrics(run_dir):
    try:
        df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
        return df
    except Exception:
        return None

metrics = {k: load_metrics(v) for k, v in runs.items()}

print("=== FINAL METRICS ===")
for k, df in metrics.items():
    if df is not None:
        final_acc = df["test_acc"].iloc[-1]
        final_asr = df["asr"].iloc[-1] if "asr" in df.columns else 0.0
        peak_asr = df["asr"].max() if "asr" in df.columns else 0.0
        print(f"{k.upper()}: Clean Acc={final_acc:.2f}%, Final ASR={final_asr:.2f}%, Peak ASR={peak_asr:.2f}%")
    else:
        print(f"{k.upper()}: Not Found")
        
print("\n=== ATTACK IMPACT ===")
if metrics["no_attack"] is not None:
    na_acc = metrics["no_attack"]["test_acc"].iloc[-1]
    na_asr = metrics["no_attack"]["asr"].iloc[-1] if "asr" in metrics["no_attack"].columns else 0.0
    for k in ["early", "middle", "late"]:
        if metrics[k] is not None:
            acc = metrics[k]["test_acc"].iloc[-1]
            asr = metrics[k]["asr"].iloc[-1]
            print(f"{k.upper()} vs NO_ATTACK: Acc Diff={acc-na_acc:.2f}%, ASR Diff={asr-na_asr:.2f}%")

print("\n=== POST-RUN EXPOSURE VALIDATION ===")
for k in ["early", "middle", "late"]:
    meta_path = os.path.join(runs[k], "attack_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            # The metadata only tells us total poisoned samples per round (250).
            # The start and end bounds are also stored.
            print(f"{k.upper()}: start={meta['attack_start_round']}, end={meta['attack_end_round']}, samples_per_round={meta['num_poisoned_samples']}")

