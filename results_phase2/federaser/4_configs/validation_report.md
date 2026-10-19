# FedEraser Configurations Validation Report

## 1. Explicit Configurations Overview
The four evaluated configurations are explicitly represented by combinations of `IID` / `Non-IID` with $E_{cal} \in \{1, 2\}$. All were executed strictly under $\Delta t = 1$, corresponding to retention across every sequential logged checkpoint (rounds 1 through 50). This represents a distinct architectural mapping diverging entirely from the earlier simulated $\Delta t=2$ constraints.
Note: Based on the raw parameters, `$E_{cal}=1$` is formally matching the calibration ratio established by $E_{loc}=2$ natively ($r = 0.5$).

## 2. Global L2-Norm Assertions
A strict verification boundary was hard-coded in `src/unlearning/federaser.py`. Before resolving any calibrated state, the exact global L2 norm of the resultant unlearning tensor must match the raw global norm of the historical update block. This check operates purely on trainable flattened tensors simultaneously, effectively ensuring magnitude precision strictly mirroring the literature.

## 3. Pre-Trained Policy Declarations
The BatchNorm bypass policy was locked and statically enforced. Integer counters (`num_batches_tracked`) alongside statistical trackers (`running_var`, `running_mean`) actively supersede FedEraser L2 attenuation blocks. They are explicitly replicated (cloned) identically from valid unlearned trajectories, shielding stability logic inherently designed by standard PyTorch aggregation.

