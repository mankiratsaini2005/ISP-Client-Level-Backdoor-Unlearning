# Methods, Metrics and Interpretation

## Methods compared

### Direct historical-update subtraction (M_sub)

The implemented baseline subtracts C0's stored historical contribution. This is a simple comparison point, not proof that the model is mathematically equivalent to a model trained without C0.

### Knowledge distillation (M_KD)

The student starts from M_sub. M_attack is used as the teacher and proxy CIFAR-100 examples are used for distillation. This is an empirical recovery procedure; it does not guarantee that the teacher's backdoor influence is absent.

### FedEraser-style reconstruction

The corrected runs use retained clients C1–C9 and calibration epochs (Ecal = 1 or 2) to reconstruct the trajectory without C0. The archived implementation scales using a global L2 norm across eligible weight/bias tensors. This is the project's implementation interpretation and must not be described as the only tensor-granularity interpretation specified by the original paper.

### Counterfactual reference (M_CF)

M_CF is trained separately through the rounds without C0, retaining C1–C9. It is a reference for comparison only, not an unlearning algorithm or guaranteed theoretical upper bound.

## Metrics

- Clean accuracy: accuracy on all 10,000 clean test images.
- ASR/TTR: percentage of the 9,000 triggered non-Airplane examples classified as target class Airplane (class 0). Lower is better for trigger suppression.
- UTR: percentage of the 9,000 untriggered non-Airplane examples predicted as Airplane (class 0). This measures natural false-target predictions.
- Airplane recall: recall on the 1,000 true Airplane test examples (class 0).
- Macro-F1: macro-average F1 across the 10 CIFAR-10 classes.
- Counterfactual agreement: percentage of examples for which a candidate model's argmax prediction agrees with M_CF on the relevant clean or triggered evaluation population. This is prediction agreement, not accuracy, a security guarantee, or proof that all training influence was erased.

## Phase 1 interpretation

In IID, the clean model has 75.11% clean accuracy and 2.73% TTR; the attacked model has 75.01% accuracy and 56.07% TTR. Accuracy changes by -0.10 percentage points while TTR rises by 53.34 points.

In the saved Non-IID split, clean accuracy changes from 70.10% to 69.80% (-0.30 percentage points), while TTR changes from 3.30% to 32.21% (+28.91 points).

Clean accuracy alone is therefore insufficient to detect this backdoor in these runs.

## Phase 2 interpretation

- M_sub achieves low TTR (0.24% IID and 0.87% Non-IID) but harms utility, especially in the saved Non-IID run.
- In the saved Non-IID M_sub run, Cat, Deer, Horse and Truck each had 0% clean test accuracy and zero predicted labels. This is a property of this run, not a universal behavior claim.
- M_KD and FedEraser recover much of the lost utility.
- Ecal=2 is a strong observed balance among the tested FedEraser settings. It is close to, but not identical to, M_CF on all metrics.
- Non-IID evidence is one saved partition / seed. Multiple partition draws and seeds are needed before making broad claims.

## Reporting rules

Always show clean utility beside ASR/TTR. Include UTR and class-sensitive metrics where space permits. Keep results partition-specific. Do not collapse security and utility into a fabricated single score. Do not claim statistical significance, causal erasure, guaranteed unlearning, or proven research novelty.
