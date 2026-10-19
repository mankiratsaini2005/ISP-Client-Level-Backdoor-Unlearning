# FedEraser Run Registry

The following existing FedEraser baseline runs are explicitly superseded:

1. **`iid/` and `noniid/`**: Superseded due to the erroneous inclusion of `weight_decay=5e-4` inherited from a deprecated baseline script (the original Phase-1 training strictly used `0.0`). Additionally, these used an incorrect evaluation function for UTR.
2. **`iid_Ecal1/`, `iid_Ecal2/`, `noniid_Ecal1/`, and `noniid_Ecal2/`**: Superseded due to the use of unseeded calibration dataloaders during local epoch training, and a faulty BatchNorm handling policy where integer `num_batches_tracked` counters erroneously escalated through calibration operations instead of being rigidly preserved.

**New Corrected Runs**:
These configurations have been regenerated under strict tracking:
- `iid_Ecal1_v2/`
- `iid_Ecal2_v2/`
- `noniid_Ecal1_v2/`
- `noniid_Ecal2_v2/`
