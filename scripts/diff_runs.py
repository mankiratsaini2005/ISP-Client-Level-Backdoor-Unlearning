import pandas as pd
df1 = pd.read_csv("results_phase2/federaser/4_configs/federaser_extended_comparison.csv")
df2 = pd.read_csv("results_phase2/federaser/corrected_evaluation_v2/federaser_comparison.csv")

metrics = ["Clean_Accuracy", "Triggered_Rate", "Untriggered_Airplane_Rate", "Macro_F1"]
for p in ["IID", "Non-IID"]:
    for m in ["M_FedEraser_Ecal1", "M_FedEraser_Ecal2"]:
        r1 = df1[(df1["Partition"] == p) & (df1["Model"] == m)]
        r2 = df2[(df2["Partition"] == p) & (df2["Model"] == m)]
        if len(r1) == 0 or len(r2) == 0: continue
        print(f"[{p} {m}]")
        for metric in metrics:
            diff = r2[metric].values[0] - r1[metric].values[0]
            print(f"  {metric}: {r1[metric].values[0]:.2f}% -> {r2[metric].values[0]:.2f}% (Diff: {diff:+.2f}%)")
