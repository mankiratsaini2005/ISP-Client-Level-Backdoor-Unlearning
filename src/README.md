# Source code

Import the canonical source tree from the working experiment repository here. Keep implementation code intact during import; refactor only in a separate reviewed change.

Recommended layout once actual files are available:

- src/data/ — dataset loading and client partition helpers.
- src/fl/ — client training, server aggregation and round loop.
- src/attacks/ — C0 data poisoning and trigger handling.
- src/unlearning/ — M_sub, M_KD and FedEraser-style reconstruction.
- src/evaluation/ — clean, triggered, untriggered, class-sensitive and counterfactual-agreement metrics.

These are suggested destinations, not assertions that those source files currently exist. Do not reconstruct missing scripts from the documentation alone.
