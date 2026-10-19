# Team Workflow

This guide is for teammates continuing the ISP project.

## Before changing anything

1. Read README.md, docs/EXPERIMENT_PROTOCOL.md, docs/METHODS_AND_METRICS.md and docs/REPRODUCIBILITY_AND_ARTIFACTS.md.
2. Check the current branch, open pull requests and latest commit.
3. Identify whether your task changes code, experiment configuration, results, or presentation.
4. Do not mix historical presentation numbers with the current baseline snapshot.

## Branching and pull requests

Use focused branch names, for example:

- docs/update-reproducibility-audit
- exp/multiple-noniid-seeds
- feat/stable-seed-derivation
- viz/client-class-heatmap
- presentation/final-slide-polish

Open a pull request with: purpose, files changed, tests/validation performed, result artifacts, and known limitations. Do not commit directly to main for ordinary development once collaborators begin working.

## Experiment record required for every run

Record:
- unique run ID and date;
- git commit SHA;
- config file path;
- dataset split / partition ID;
- Python, PyTorch, CUDA and hardware details;
- global rounds, local epochs, optimizer and batch size;
- random seed policy and realized seed values;
- attack client, attack rounds, poison rate, trigger and target class;
- output directory and checkpoint/delta filenames;
- metric script and exact evaluation denominators;
- runtime boundaries, including what the timer excludes.

## Results policy

- Store compact, validated CSV/JSON summaries in Git.
- Do not overwrite a prior result silently. Add a new run ID/version and record whether it supersedes an earlier run.
- Preserve full precision in machine-readable CSV where it exists; use rounded numbers only for slides.
- Mark approximate metrics and missing measurements explicitly.
- Plot from committed CSVs rather than hand-entered spreadsheet cells whenever possible.
- Treat the current Non-IID results as one saved partition / seed.
- Do not claim statistical significance or broad robustness without the appropriate repeated experiments.

## Large files and data

Do not commit raw CIFAR-10 archives, large model weights, or generated caches directly without team agreement. Use Git LFS or approved institutional storage where needed and add a manifest with file name, size, hash, and storage location. Never commit credentials, tokens, .env files or private user information.

## Presentation workflow

Keep a canonical editable PPTX and its source slide-content Markdown together. Exported PDFs are useful previews but should not replace the editable source. Put older decks in a clearly named archive folder and label them with their protocol/result snapshot. Speaker notes, Q&A preparation and figures should be traceable to the same metric CSVs.
