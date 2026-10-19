# Reproducibility and Artifact Audit

## What is documented here

This repository's initial project handoff contains the current experimental protocol and summarized metric CSVs transcribed from the project's recorded baseline results. These summaries are useful for reporting, but are not a substitute for raw run artifacts.

## Required source artifacts still to import

- Training source code and evaluation scripts.
- Exact config files and command lines for each run.
- Client partition metadata and class-count histograms.
- Saved per-round client updates / deltas and server checkpoints needed for reconstruction.
- Raw and parsed evaluation logs.
- Run registry and seed manifest.
- SHA-256 checksums for configs, scripts, result files, and retained key artifacts.
- Source/editable PowerPoint used for the final presentation, plus speaker notes and validated figures.

Until these files are imported and cross-checked, describe the CSVs as a recorded result snapshot; do not claim this repository by itself is a one-command reproduction package.

## FedEraser corrected-run record

Corrected run locations in the prior working tree were recorded as:

- results_phase2/federaser/iid_Ecal1_v2/
- results_phase2/federaser/iid_Ecal2_v2/
- results_phase2/federaser/noniid_Ecal1_v2/
- results_phase2/federaser/noniid_Ecal2_v2/

Recorded settings:
- dt = 1, rounds 1–50, Ecal in {1, 2}; reconstruction round 1 bypasses calibration.
- SGD learning rate 0.01, momentum 0.9, batch size 64, explicit weight decay = 0.
- BatchNorm running_mean / running_var pass through unscaled.
- num_batches_tracked is excluded from scaling and remains unchanged.
- Update scaling uses global L2 norm across eligible weight/bias tensors in the archived implementation.

## Seed reproducibility caveat

The recorded client/round seed expression was:

    hash((42, round_num, c)) % (2**32)

It was used to seed Python random, NumPy, Torch, CUDA (when available), and the DataLoader generator. Python's built-in hash can be randomized between processes unless PYTHONHASHSEED is controlled. Therefore, do not promise cross-process deterministic results based on this expression alone. Resolve this by replacing it with a stable seed derivation (for example, a documented deterministic hash or SeedSequence design), then rerun or explicitly establish whether the exact saved run can be reproduced.

Do not silently change the seed policy and then present resulting numbers as the exact same historical run.

## Timing boundaries

Historical M_CF full-run timers include repeated in-loop evaluation over clean, triggered and untriggered populations (about 28,000 examples over three evaluation passes per round), final evaluation, and potentially saving depending on the timer boundary.

FedEraser reconstruction-loop times include saved-delta file reads but are not end-to-end training times. The controlled local-compute benchmark excludes global aggregation/update and evaluation. These records must not be compared as a single speed ranking.

## Audit checklist

- [ ] Import source scripts and configs from the canonical experiment working tree.
- [ ] Verify result CSV values against saved logs and metric denominators.
- [ ] Verify every run's partition ID, seed, attack configuration, and model checkpoint.
- [ ] Fix or pin the FedEraser seed policy and document the decision.
- [ ] Calculate checksums and store a manifest.
- [ ] Re-run a small smoke test before any expensive run.
- [ ] Reproduce baseline metrics before starting new method variants.
- [ ] Keep new experiments on dedicated branches and record run IDs.
