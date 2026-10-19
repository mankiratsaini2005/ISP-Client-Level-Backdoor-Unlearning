# Phase 2: Final Validation and Findings

## 1. Reproducible $M_{CF}$ Aggregation Audit
An audit verified the theoretical and empirical weighting behavior of the `M_CF` (Counterfactual) model in the Non-IID setting. The `client_partitions.pt` tensor shows that by filtering out Client 0, the remaining nine clients (C1–C9) supplied a combined total of `46,403` training samples. 

This directly matches the explicit source implementation in `scripts/run_noniid_m_cf.py`:
```python
        # OMIT CLIENT 0
        for client_id in range(1, config["num_clients"]):
            # [local training]
            client_deltas.append(delta)
            sample_counts.append(len(partitions[client_id]))
            
        assert len(client_deltas) == 9
        
        # FedAvg naturally renormalizes via sum(sample_counts)
        global_state = aggregate_fedavg(global_state, client_deltas, sample_counts)
```
The `aggregate_fedavg` call correctly derives its denominator by summing `sample_counts`, yielding $46{,}403$. The empirical FedAvg scaling factors calculated per client ($n_k / 46{,}403$) are:
- **C1-C9 Sample counts**: `[5704, 2879, 2857, 6512, 4084, 7936, 4043, 3670, 8718]`
- **C1-C9 Aggregation weights**: `[0.122923, 0.062043, 0.061569, 0.140336, 0.088012, 0.171023, 0.087128, 0.079090, 0.187876]`
- **Sum of weights**: `1.000000`

## 2. Characterization of Non-IID $M_{sub}$ Performance Drop
In contrast to the targeted accuracy drop observed in the IID setting, evaluating the Non-IID historical subtraction model ($M_{sub}$) shows broad performance reductions across multiple classes.
- **Per-Class Clean Accuracy**: 
  - `0 (Airplane)`: 14.2%
  - `1 (Automobile)`: 13.5%
  - `2 (Bird)`: 59.2%
  - `3 (Cat)`: 0.0%
  - `4 (Deer)`: 0.0%
  - `5 (Dog)`: 3.5%
  - `6 (Frog)`: 91.1%
  - `7 (Horse)`: 0.0%
  - `8 (Ship)`: 98.9%
  - `9 (Truck)`: 0.0%
- **Predicted Class Distribution (Test 10k)**:
  - `{0: 234, 1: 251, 2: 2424, 3: 0, 4: 0, 5: 42, 6: 2901, 7: 0, 8: 4148, 9: 0}`

Four classes generated zero predictions across the entire 10,000-image test set, resulting in a Macro F1 score of 17.1%.

## 3. Teacher Predictions on the Proxy Data
To record the output behavior of the attacked models prior to KD training, we ran inference on the fixed 5,000-image CIFAR-100 proxy. The models output predictions across all 10 target classes. 
- **IID Teacher**: `{0: 633, 1: 178, 2: 785, 3: 695, 4: 446, 5: 502, 6: 763, 7: 313, 8: 363, 9: 322}` (Total: 5,000)
- **Non-IID Teacher**: `{0: 634, 1: 166, 2: 935, 3: 301, 4: 203, 5: 789, 6: 797, 7: 227, 8: 375, 9: 573}` (Total: 5,000)
*Note: These counts merely record the baseline behavior of the models prior to training. They do not demonstrate identical distributions nor prove an absence of class bias.*

## 4. Evaluation Row Verification
The final audit script dynamically loaded the original Non-IID evaluation JSON and the new Non-IID evaluation JSON (generated after completing the KD runs). It programmatically compared the following metrics for the `Logged Non-IID Attack`, `Non-IID M_sub`, and `Non-IID M_CF` models: `Clean_Acc`, `Clean_Loss`, `Macro_F1`, `Triggered_Rate`, `Untriggered_Airplane_Rate`, `Airplane_Recall`, `CF_Agreement_Clean`, and `CF_Agreement_Triggered`. 

- **Result**: Every metric matched within a tolerance of `1e-4`, and no fields were missing.
- **Status**: **PASS**

## 5. Proxy Provenance & Integrity
- **Source**: The `cifar100` dataset was sourced from Hugging Face (`uoft-cs/cifar100`). 
- **Integrity**: Because it was downloaded and formatted differently than the standard torchvision archive, the MD5 integrity check in the torchvision source code was bypassed via a lambda patch (`cifar.check_integrity = lambda path, md5=None: True`). 
- **Data subset**: `50` images per class (from all 100 classes) were deterministically selected using seed `42`.
- **SHA-256 (Raw 5,000-image subset tensor)**: `630dc3286c8dd16a74a8ac4d6c7ac92fa59dc6868d5a6c6a77b793585f79d4c7`.
- **Label Usage**: The CIFAR-100 labels were used to ensure a balanced 50/class selection ratio. They were not used in the Knowledge Distillation loss function.

---

## 6. Consolidated Results Table

### Knowledge Distillation Setup
The hyperparameter settings were retained exactly from a Phase-1 historical archive script without further tuning or adjustment. 
- **Loss**: Pure $T^2$-scaled KL Divergence ($T=4$)
- **Optimizer**: SGD (LR: 0.001, Momentum: 0.9)
- **Epochs**: 5
- **Batch Size**: 64
- **Architecture**: Teacher set to `eval()`. Student set to `train()` but with all `BatchNorm2d` running statistics frozen (bit-identical before and after KD) and Dropout active.

### Metrics Table

| Partition | Model | Clean Acc | ASR / TTR | Clean CF Agree | Air. Recall | Macro F1 | Sub. Time (s) | KD Time (s) | Total Unlearning (s) | Retrain $M_{CF}$ (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IID** | Phase-1 Clean | 75.11% | 2.73% | 93.01% | 80.5% | 74.9% | - | - | - | - |
| **IID** | Logged Attack | 75.01% | 56.07% | 92.77% | 82.0% | 74.8% | - | - | - | - |
| **IID** | $M_{sub}$ | 66.88% | 0.24% | 76.25% | 40.6% | 66.8% | 0.18s | - | 0.18s | - |
| **IID** | $M_{KD}$ | 72.82% | 2.40% | 88.54% | 81.4% | 72.7% | - | 7.64s | 7.82s | - |
| **IID** | $M_{CF}$ *(Ref)* | 74.87% | 3.01% | - | 82.4% | 74.7% | - | - | - | 2513.87s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Non-IID** | Phase-1 Clean | 70.10% | 3.30% | 90.21% | 80.9% | 69.0% | - | - | - | - |
| **Non-IID** | Logged Attack | 69.80% | 32.21% | 90.28% | 81.3% | 68.7% | - | - | - | - |
| **Non-IID** | $M_{sub}$ | 28.04% | 0.87% | 35.31% | 14.2% | 17.1% | 3.77s | - | 3.77s | - |
| **Non-IID** | $M_{KD}$ | 60.36% | 3.69% | 78.65% | 75.6% | 57.7% | - | 7.52s | 11.29s | - |
| **Non-IID** | $M_{CF}$ *(Ref)* | 69.64% | 3.40% | - | 79.0% | 68.4% | - | - | - | 4219.17s |

*Limitations Note: Knowledge Distillation improves the apparent performance metrics without demonstrating active exploitation of the backdoor trigger on this test set. However, these metrics do not confirm full recovery, and do not guarantee an absence of backdoor re-emergence if the model is subjected to further fine-tuning or novel triggers.*

## 7. Version Control and File Handling
1. A `.gitignore` file successfully limits tracking in the repository, explicitly excluding large memory artifacts and untracked images. **The `trajectory/` directories (1.1GB per attack) and the `client_partitions.pt` tensors (211MB per setting) are NOT committed to either tag.** Untracked historical markdown documents in `docs/` and untracked Phase-1 image files (`results/per_class_asr.png`) were also omitted from the repository entirely.
2. **Phase 1 Commit (`phase1-frozen`):** Contains the original source code, configuration files, and standard FL evaluation outputs.
3. **Phase 2 Commit (`phase2-subtraction-kd-frozen`):** Contains the newly generated Phase 2 unlearning evaluations, JSON configuration dictionaries, python scripts, `sha256_manifest.txt`, charts, and this final report.

### Output Charts
The following comparative diagnostic charts have been generated and saved locally:
1. `results_phase2/charts/clean_acc.png` — Clean Accuracy
2. `results_phase2/charts/clean_cf.png` — Clean Counterfactual Agreement. *(M_CF is omitted from the bars because its self-agreement is trivially 100%; instead, its natural metrics serve strictly as a reference in the data table.)*
3. `results_phase2/charts/airplane_recall.png` — Target-class Recall (Airplane)
4. `results_phase2/charts/kd_loss.png` — $T^2$-scaled KL Divergence Loss across KD Epochs

### Model SHA-256 Manifest
The 51-file manifest `sha256_manifest.txt` enumerates hashes for all Phase 1 and Phase 2 models, `.json` files, and `cifar100_proxy_indices.json`. Zero files were flagged missing.
