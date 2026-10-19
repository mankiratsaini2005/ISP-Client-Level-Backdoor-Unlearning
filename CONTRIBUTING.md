# Contributing

Thanks for helping continue this ISP project.

## Pull request expectations

- Keep one PR focused on one task.
- Explain the motivation and expected outcome.
- Link any issue or run ID where available.
- Include validation steps and exact commands.
- Include before/after figures for chart changes when useful.
- Keep existing result snapshots immutable unless there is a verified correction; add a changelog note when a result is superseded.

## Scientific integrity

Do not fabricate experimental results, class counts, uncertainty intervals, citations, or research gaps. A schematic must be labelled schematic. A recorded result that cannot yet be reproduced from repository contents must be labelled as such. Distinguish observations from hypotheses.

## Code and data hygiene

- Do not commit secrets or environment files.
- Avoid committing raw datasets and large model checkpoints without approval.
- Use deterministic, documented random-number generation.
- Keep metric denominators and class mappings explicit.
- Check that clean accuracy, ASR/TTR, UTR, recall, Macro-F1 and counterfactual agreement are not conflated.
