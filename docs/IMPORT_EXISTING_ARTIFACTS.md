# Import the existing working-tree artifacts

The GitHub repository is created and the documentation/result snapshot is available on the docs/team-handoff branch. This page explains how to bring over the actual experiment source and presentation files without accidentally publishing the wrong draft or overwriting history.

## 1. Fetch the handoff branch

    git clone https://github.com/mankiratsaini2005/ISP-Client-Level-Backdoor-Unlearning.git
    cd ISP-Client-Level-Backdoor-Unlearning
    git checkout docs/team-handoff

## 2. Import the canonical experiment source

Copy the real training source, config files, scripts, run registry, seed manifest, and saved evaluation logs from the authoritative project working tree into the repository. Preserve original paths or document any intentional path mapping. Do not generate replacement training code from README descriptions.

Before copying raw outputs, check sizes and remove local caches, temporary files, credentials, private records, and any data that should not be shared. Do not commit the raw CIFAR-10 dataset or large checkpoints directly without team agreement. Use Git LFS or approved storage for large files.

## 3. Import presentation materials

There are several historic/in-progress deck families in the work folder. The 12-slide mid-semester deck notes contain an older 20-client / 100-round setup and earlier results; these values must not silently be combined with the current 10-client / 50-round CSV snapshot.

Keep older materials under presentation/archive/ with a README identifying their protocol and status. Store the canonical editable PowerPoint, its source Markdown, speaker notes, presentation PDF preview, and figures together once the team confirms which version is final. Quantitative figures must match the selected result CSVs.

## 4. Commit imported files

    git status --short
    git diff --stat
    git add src scripts configs manifests presentation
    git commit -m "Import canonical experiment and presentation artifacts"
    git push origin docs/team-handoff

Review the diff before committing. Do not use git add -A blindly if the working tree contains datasets, model checkpoints or unrelated files.

## 5. Validate before claiming reproducibility

- Compare each result CSV with raw run logs and checkpoints.
- Verify metric denominators, especially ASR/TTR and UTR over 9,000 examples each.
- Identify the exact Non-IID partition draw and seed.
- Resolve the Python hash() seed caveat described in docs/REPRODUCIBILITY_AND_ARTIFACTS.md.
- Generate checksums and update the run registry.
- Reproduce a small smoke run before running large experiments.

The open pull request updates automatically when more commits are pushed to docs/team-handoff.
