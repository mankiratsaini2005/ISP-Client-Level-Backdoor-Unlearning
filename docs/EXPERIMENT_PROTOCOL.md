# Experimental Protocol — Current Baseline Snapshot

This document captures the latest baseline setup recorded in the project handoff. Verify all settings against the canonical configs and run logs after they are imported.

## Task and question

Can we reduce the influence of a malicious federated client, suppress its backdoor, and preserve clean-model utility? Evaluate security and utility jointly, and compare predictions with a model independently trained without the malicious client.

## Dataset and model

- Dataset: CIFAR-10.
- Standard split recorded: 50,000 training images and 10,000 clean test images.
- Model: SmallCNN.
- Number of clients: 10 (C0–C9).
- Participation: all 10 clients participate in every main clean/attack round.
- Rounds: 50.
- Local epochs per round: 2.
- Optimizer: SGD.
- Learning rate: 0.01.
- Momentum: 0.9.
- Batch size: 64.
- Project seed recorded: 42.

## Data partitioning

Two client-data regimes are recorded:

1. IID partition.
2. Dirichlet Non-IID partition with alpha = 0.5.

The current Non-IID results use a single saved partition draw / seed. Report the partition-specific results as one-run evidence; do not imply multi-seed stability or generalization to all Dirichlet draws.

## Backdoor specification

- Malicious client: C0.
- Attack activation: rounds 30–50 inclusive. Rounds 1–29 are pre-attack.
- Poisoning rate: 10% of C0's local data.
- Trigger: 3×3 white patch in the bottom-right corner.
- Target label: Airplane, CIFAR-10 class index 0.
- IID C0 population: 5,000 local samples; approximately 500 poisoned.
- Saved Non-IID C0 population: 3,597 local samples; approximately 359 poisoned; this is 7.194% of the full 50,000-image training population.

## Model endpoints

- M_clean: clean-training baseline without the backdoor attack.
- M_attack: endpoint of attacked training.
- M_sub: implemented direct-subtraction baseline using C0's stored historical contribution.
- M_KD: student initialized from M_sub, with M_attack as teacher and proxy CIFAR-100 data for knowledge distillation.
- FedEraser Ecal=1 / Ecal=2: corrected FedEraser-style reconstruction settings using retained clients C1–C9 and one or two calibration epochs.
- M_CF: independent counterfactual training branch run without C0, retaining C1–C9. It is a reference, not an unlearning method.

## Evaluation populations

- Clean CIFAR-10 test set: 10,000.
- Triggered non-Airplane test examples: 9,000.
- Untriggered non-Airplane test examples: 9,000.
- True Airplane class examples: 1,000.

The triggered and untriggered 9,000-image populations are two versions of the same non-Airplane examples. These are related evaluation views, not four disjoint groups.

## Current scope boundaries

This snapshot is deliberately limited to CIFAR-10 + SmallCNN and the recorded IID / single saved Non-IID split. Do not silently mix in values from earlier mid-semester presentation drafts; some earlier slide-generation notes describe an older protocol (20 clients, 100 rounds) and earlier metric values. Historical slide versions should be archived and labelled with their protocol, not treated as the canonical current results.
