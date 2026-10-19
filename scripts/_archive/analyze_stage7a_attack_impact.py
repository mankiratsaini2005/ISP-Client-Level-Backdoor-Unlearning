import os
import glob
import pandas as pd
import json

base_dir = "results"
dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("stage7a_") and os.path.isdir(os.path.join(base_dir, d))])

runs = {}
for d in dirs:
    parts = d.split('_')
    # stage7a_c1_attacked_20260910...
    if len(parts) >= 4:
        c_idx = parts[1] # c1, c2, c3, c4
        rtype = parts[2] # attacked, noattack, mcf
        
        csv_path = os.path.join(base_dir, d, "metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # The metrics.csv has cols like 'round', 'test_loss', 'test_acc', 'asr'
            if 'test_acc' in df.columns:
                best_clean = df['test_acc'].max()
                final_clean = df.iloc[-1]['test_acc']
            elif 'clean_acc' in df.columns:
                best_clean = df['clean_acc'].max()
                final_clean = df.iloc[-1]['clean_acc']
            else:
                continue
                
            if 'asr' in df.columns:
                peak_asr = df['asr'].max()
                final_asr = df.iloc[-1]['asr']
            else:
                peak_asr = 0
                final_asr = 0
                
            runs[(c_idx, rtype)] = {
                'dir': d,
                'final_clean': final_clean,
                'best_clean': best_clean,
                'peak_asr': peak_asr,
                'final_asr': final_asr
            }

print("=== STAGE 7A ATTACK-VALIDITY CHECK ===")

for c in ['c1', 'c2', 'c3', 'c4']:
    attacked = runs.get((c, 'attacked'))
    if c == 'c4':
        noattack = runs.get(('c3', 'noattack')) # C4 reuses C3 noattack
    else:
        noattack = runs.get((c, 'noattack'))
        
    if attacked and noattack:
        acc_drop = noattack['final_clean'] - attacked['final_clean']
        print(f"\nCondition {c.upper()}:")
        print(f"  No-Attack Clean Acc : {noattack['final_clean']:.2f}% (Best: {noattack['best_clean']:.2f}%)")
        print(f"  Attacked Clean Acc  : {attacked['final_clean']:.2f}% (Best: {attacked['best_clean']:.2f}%) -> Drop: {acc_drop:+.2f}%")
        print(f"  Peak ASR            : {attacked['peak_asr']:.2f}%")
        print(f"  Final ASR           : {attacked['final_asr']:.2f}%")
        
        if attacked['peak_asr'] < 10.0:
            print(f"  [!] WARNING: Very low backdoor insertion.")
        else:
            print(f"  [x] Attack valid.")
