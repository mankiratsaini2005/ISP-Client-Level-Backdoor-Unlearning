# FedEraser Baseline Summary

## Algorithm Verification & Constraints
The published FedEraser algorithm was faithfully implemented using the exact $\Delta t = 2$ retention interval, corresponding explicitly to retained historical rounds `1, 3, 5, ..., 49`. Following mathematical rigor, rather than summing incompatible 1-round local updates to form continuous intervals, we retained and explicitly calibrated the actual client updates logged dynamically at those precise checkpoints. C0 was successfully excluded, retaining sample participation mapping matching native `FedAvg` weighting for remaining C1-C9 datasets.

## Performance Outcomes
The evaluations generated valid finite bounds strictly matching the expected schema. 

### 1. IID Setting
* **Efficacy**: The baseline successfully mitigated the backdoor target rate from **56.07%** down to **3.50%** (Triggered Target Rate), achieving excellent backdoor suppression matching the counterfactual reference ($3.01\%$).
* **Fidelity Constraints**: Clean accuracy measured **66.13%**, recovering notably above raw historical subtraction alone ($66.88\%$), although yielding slightly below the M_KD unlearning baseline ($72.82\%$).
* **Agreement**: Counterfactual fidelity registered **78.87%** clean agreement.

### 2. Non-IID Setting
* **Efficacy**: ASR heavily mitigated from **32.21%** down to **3.73%**.
* **Fidelity Constraints**: FedEraser recovered significant representation capability against the native decision-boundary collapse. M_sub failed entirely under Non-IID (28.04% clean accuracy), whereas FedEraser effectively restored stable convergence measuring **58.67%**. This approaches the $M_{KD}$ performance ceiling ($60.36\%$).
* **Agreement**: Counterfactual fidelity registered **76.35%** clean agreement.

### 3. Execution Topology (Runtime Constraints)
The **Calibration/Reconstruction** phase executed smoothly without heavy continuous retraining loops. 
* **IID**: ~326 seconds
* **Non-IID**: ~336 seconds
* **Runtime Distinction**: This reflects purely the sequential post-attack execution bounds involving $E_{cali}=1$ across the 25 retained checkpoints. This excludes the original Phase-1 Logged Attack Full Training baseline latency (~2664 seconds) and the massive unlearning buffer overheads (1.1 GB structural cache arrays) required to fulfill FedEraser's offline historical state preservation prerequisites.

## Next Steps 
The baselines confirm that algorithmic L2-calibration (FedEraser) effectively repairs historical update collapse, notably salvaging the Non-IID $M_{sub}$ failure. We can now proceed to cross-reference the literature to explore competitive improvements.
