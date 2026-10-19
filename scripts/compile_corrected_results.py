import os
import json
import pandas as pd

out_dir = "results_phase2/federaser/corrected_evaluation"
os.makedirs(out_dir, exist_ok=True)

def load_json(path):
    if not os.path.exists(path): return {}
    with open(path) as f: return json.load(f)

# Load existing tables from Phase 1 and 2
iid_clean_p1 = load_json("results/iid_clean/final_results.json")
noniid_clean_p1 = load_json("results/noniid_clean/final_results.json")
iid_eval = load_json("results_phase2/phase2_kd_evaluation.json")
noniid_eval = load_json("results_phase2/noniid_kd_evaluation.json")

# Load corrected FedEraser results
iid_federaser = load_json(f"{out_dir}/iid_corrected_evaluation.json")
noniid_federaser = load_json(f"{out_dir}/noniid_corrected_evaluation.json")

def get_rt(path, key="runtime_seconds"):
    d = load_json(path)
    return d.get(key) if d.get(key) is not None else d.get("total_unlearning_runtime", d.get("subtraction_time_seconds"))

table_data = []

def create_row(partition, model, base_dict, eval_dict, rt_label, rt_val):
    row = {"Partition": partition, "Model": model}
    
    row["Clean_Accuracy"] = base_dict.get("Clean_Acc", base_dict.get("clean_accuracy", base_dict.get("Clean_Accuracy")))
    row["Clean_Loss"] = base_dict.get("Clean_Loss", base_dict.get("clean_loss"))
    row["Triggered_Rate"] = base_dict.get("Triggered_Rate", base_dict.get("triggered_rate"))
    row["Untriggered_Airplane_Rate"] = base_dict.get("Untriggered_Airplane_Rate", base_dict.get("untriggered_airplane_rate"))
    
    f1 = base_dict.get("Macro_F1", base_dict.get("clean_macro_f1"))
    if f1 is not None and f1 <= 1.0: f1 = f1 * 100.0
    row["Macro_F1"] = f1
    
    if "M_CF" in model:
        row["Clean_CF_Agreement"] = "Reference"
        row["Triggered_CF_Agreement"] = "Reference"
    else:
        c_cf = eval_dict.get("CF_Agreement_Clean", eval_dict.get("Clean_CF_Agreement"))
        t_cf = eval_dict.get("CF_Agreement_Triggered", eval_dict.get("Triggered_CF_Agreement"))
        row["Clean_CF_Agreement"] = c_cf if c_cf is not None else "N/A"
        row["Triggered_CF_Agreement"] = t_cf if t_cf is not None else "N/A"
        
    ar = eval_dict.get("Airplane_Recall")
    if ar is None and "Per_Class_Clean_Accuracy" in eval_dict:
        ar = eval_dict["Per_Class_Clean_Accuracy"].get("0")
    row["Airplane_Recall"] = ar if ar is not None else "N/A"
    
    row["Runtime_Type"] = rt_label
    row["Runtime_Seconds"] = rt_val if rt_val is not None else "N/A"
        
    return row

iid_kd_rt = get_rt("results_phase2/iid_kd/metadata.json", "kd_runtime_seconds")
iid_sub_rt = get_rt("results_phase2/iid_subtraction/subtraction_runtime.json")
iid_tot_kd = (iid_sub_rt + iid_kd_rt) if (iid_sub_rt is not None and iid_kd_rt is not None) else None

table_data.append(create_row("IID", "Phase-1 Clean", iid_clean_p1, iid_eval.get("Phase-1 Clean", {}), "Full Training", get_rt("results/iid_clean/final_results.json")))
table_data.append(create_row("IID", "Attack (No unlearning)", iid_eval.get("Logged IID Attack", {}), iid_eval.get("Logged IID Attack", {}), "Logged Training", get_rt("results_phase2/iid_attack_logged/runtime.json")))
table_data.append(create_row("IID", "M_sub", iid_eval.get("M_sub", {}), iid_eval.get("M_sub", {}), "Subtraction-only", iid_sub_rt))
table_data.append(create_row("IID", "M_KD", iid_eval.get("M_KD", {}), iid_eval.get("M_KD", {}), "Subtraction+KD", iid_tot_kd))
table_data.append(create_row("IID", "M_FedEraser", iid_federaser, iid_federaser, "Calibration/Reconstruction", get_rt(f"results_phase2/federaser/iid/config.json", "reconstruction_time_seconds")))
table_data.append(create_row("IID", "M_CF", iid_eval.get("M_CF", {}), iid_eval.get("M_CF", {}), "Retraining", get_rt("results_phase2/iid_counterfactual/final_results.json")))

niid_kd_rt = get_rt("results_phase2/noniid_kd/metadata.json", "kd_runtime_seconds")
niid_sub_rt = get_rt("results_phase2/noniid_subtraction/subtraction_runtime.json")
niid_tot_kd = (niid_sub_rt + niid_kd_rt) if (niid_sub_rt is not None and niid_kd_rt is not None) else None

table_data.append(create_row("Non-IID", "Phase-1 Clean", noniid_clean_p1, noniid_eval.get("Phase-1 Non-IID Clean", {}), "Full Training", get_rt("results/noniid_clean/final_results.json")))
table_data.append(create_row("Non-IID", "Attack (No unlearning)", noniid_eval.get("Logged Non-IID Attack", {}), noniid_eval.get("Logged Non-IID Attack", {}), "Logged Training", get_rt("results_phase2/noniid_attack_logged/runtime.json")))
table_data.append(create_row("Non-IID", "M_sub", noniid_eval.get("Non-IID M_sub", {}), noniid_eval.get("Non-IID M_sub", {}), "Subtraction-only", niid_sub_rt))
table_data.append(create_row("Non-IID", "M_KD", noniid_eval.get("Non-IID M_KD", {}), noniid_eval.get("Non-IID M_KD", {}), "Subtraction+KD", niid_tot_kd))
table_data.append(create_row("Non-IID", "M_FedEraser", noniid_federaser, noniid_federaser, "Calibration/Reconstruction", get_rt(f"results_phase2/federaser/noniid/config.json", "reconstruction_time_seconds")))
table_data.append(create_row("Non-IID", "M_CF", noniid_eval.get("Non-IID M_CF", {}), noniid_eval.get("Non-IID M_CF", {}), "Retraining", get_rt("results_phase2/noniid_counterfactual/final_results.json")))

df = pd.DataFrame(table_data)
df.to_csv(f"{out_dir}/federaser_comparison.csv", index=False)

def fmt(v, is_pct=False):
    if v is None: return "N/A"
    if v == "Reference" or v == "N/A": return v
    if isinstance(v, (int, float)):
        if is_pct: return f"{v:.2f}%"
        return f"{v:.4f}"
    return str(v)

md_lines = ["| Partition | Model | Clean Acc | Clean Loss | ASR/TTR | UTR | Clean CF Agree | Triggered CF Agree | Air. Recall | Macro F1 | Runtime Type | Runtime (s) |", 
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    line = f"| {r['Partition']} | {r['Model']} | {fmt(r['Clean_Accuracy'], True)} | {fmt(r['Clean_Loss'])} | {fmt(r['Triggered_Rate'], True)} | {fmt(r['Untriggered_Airplane_Rate'], True)} | {fmt(r['Clean_CF_Agreement'], True)} | {fmt(r['Triggered_CF_Agreement'], True)} | {fmt(r['Airplane_Recall'], True)} | {fmt(r['Macro_F1'], True)} | {r['Runtime_Type']} | {fmt(r['Runtime_Seconds'])} |"
    md_lines.append(line)
with open(f"{out_dir}/federaser_comparison.md", "w") as f: f.write("\n".join(md_lines) + "\n")
print("Saved federaser_comparison.csv and federaser_comparison.md")
