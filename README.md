# ISP Project: Client-Level Backdoor Unlearning in Federated Learning

Research workspace for the ISP project **Removing Backdoors from Federated Models via Machine Unlearning**.

The current baseline question is whether the influence of one malicious federated client can be reduced while suppressing a backdoor and preserving clean-task utility. The project compares direct historical-update subtraction, subtraction followed by knowledge distillation, and FedEraser-style reconstruction. A counterfactual model trained without the malicious client is used as a reference only.

## Current status

- Baseline result records and methodology are documented in this repository.
- The current baseline snapshot is CIFAR-10 + SmallCNN, 10 clients, 50 rounds, 2 local epochs, and attack activation from round 30.
- IID and one saved Dirichlet Non-IID partition (alpha = 0.5) were evaluated. Current Non-IID results represent one partition draw / seed and must not be generalized to all non-IID settings.
- The final research contribution is not selected yet. Next step after reproducibility/artifact audit: literature review and evidence-backed research-gap selection.
- **Important:** this initial handoff commit contains documentation and summarized CSV records. The original training scripts, full run directories, partition metadata, saved deltas/checkpoints, and complete raw logs still need to be imported from the working experiment environment. Do not claim the repo alone reproduces every number until those artifacts and exact commands are added.

## Repository guide

- docs/ — experiment protocol, metrics, reproducibility caveats and team workflow.
- results/ — CSV snapshots of recorded baseline metrics and timer boundaries.
- presentation/ — presentation asset inventory and guidance; add the current editable deck and validated supporting assets here.
- configs/ — canonical run configurations should be added here when the source config files are imported.
- scripts/ and src/ — import the real training/evaluation code from the project working copy; do not recreate missing code from prose.
- manifests/ — add checksums, run registry, seed manifest and artifact provenance from the original environment.

## Start here

1. Read docs/EXPERIMENT_PROTOCOL.md.
2. Read docs/METHODS_AND_METRICS.md before interpreting a chart.
3. Read docs/REPRODUCIBILITY_AND_ARTIFACTS.md for known limitations.
4. Review results/README.md and the CSV files.
5. Follow CONTRIBUTING.md before submitting changes.

## Key result pattern

- Clean accuracy alone did not reveal the backdoor: in Phase 1, IID accuracy moved from 75.11% to 75.01% while ASR/TTR moved from 2.73% to 56.07%.
- Direct subtraction reduced ASR/TTR but caused a severe Non-IID utility failure in the saved run (28.04% clean accuracy).
- FedEraser Ecal=2 reached 74.99% accuracy / 2.34% TTR in IID and 68.89% / 3.09% in the saved Non-IID split.
- M_CF is a separate counterfactual reference trained without C0, not an unlearning method and not automatically a theoretical upper bound.

These are observations from the saved runs, not guarantees or claims of statistical significance.

## Data, artifacts and privacy

Do not commit credentials, environment files, private user data, raw datasets, or large model checkpoints by default. Check dataset/licensing rules and use Git LFS or approved storage for large artifacts. Keep derived summary CSVs and run manifests in Git when they are small and safe to share.

## Collaboration

Use a feature branch for each meaningful experiment or documentation update and open a pull request. Every reported metric should point to a run directory, config, code revision, seed/partition metadata, and evaluation script once those artifacts have been imported.
