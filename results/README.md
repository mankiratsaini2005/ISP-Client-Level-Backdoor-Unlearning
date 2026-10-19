# Recorded baseline result snapshots

These CSVs transcribe the current project result record available for this handoff. They are intended for team reference and plotting. They are not regenerated from experiment source in this initial documentation commit because the original training scripts, saved run folders and raw logs have not yet been imported here.

Before using them as final published numbers, cross-check each row against the canonical files from the experiment working tree and update the artifact manifest. Preserve the current copies as a reference snapshot; if any value changes after verification, add a dated correction note rather than silently overwriting history.

## Files

- phase1_summary.csv — clean vs attacked training.
- phase2_iid.csv — six IID endpoints and their main metrics.
- phase2_noniid.csv — six Non-IID endpoints from one saved partition / seed.
- federaser_corrected_runs.csv — corrected FedEraser Ecal=1/2 run summaries.
- runtime_ledger.csv — timer values and the work each timer includes.

## Metric denominators

- Clean accuracy: all 10,000 clean CIFAR-10 test images.
- ASR/TTR: 9,000 triggered non-Airplane images predicted as Airplane (class 0).
- UTR: 9,000 untriggered non-Airplane images predicted as Airplane (class 0).
- Airplane recall: 1,000 true Airplane images.
- Counterfactual agreement: argmax prediction agreement with M_CF on the named population; not accuracy or proof of erasure.

The two 9,000-image non-Airplane populations are alternate views (with and without trigger), not separate disjoint examples from each other.
