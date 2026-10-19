# Presentation assets and status

The presentation should have one canonical editable PowerPoint source, its slide-content/source Markdown, speaker notes, figures, and an exported PDF preview. Keep these files versioned together so teammates can trace each chart to the CSV in results/.

## Presentation variants in the working materials

There are several historical snapshots and intermediate versions in the available working materials. In particular:

- A detailed 27-slide PowerPoint named ClientLevel_Backdoor_Unlearning_FL_initial_draft.pptx covers the current Phase 1 and Phase 2 baseline result snapshot.
- A 12-slide mid-semester deck/slide-content family exists from an earlier stage. Some slide-generation notes explicitly prescribe an older setup (20 clients, 100 rounds) and earlier values such as 74.31% / 87.15% for the attacked model and 74.78% / 3.13% for a counterfactual reference. Do not silently merge these with the current 10-client, 50-round CSVs in results/.
- A separate 8-visual-slide deck and speaker guide also exist as supporting conceptual material.

## Canonicalization before upload

Before choosing what to present as the final deck:
1. Confirm with the team which editable PPTX is the current final presentation.
2. Keep intermediate versions under an archive directory, named with date/status.
3. Ensure every quantitative slide uses values from results/phase1_summary.csv and the Phase 2 CSVs, or clearly labels an earlier protocol as historical.
4. Keep a short README in every archive folder explaining the protocol/metrics represented.
5. Never replace the editable PPTX with PDF only; store the PDF as a preview alongside its source.

The actual training source code, complete raw result directories and definitive final editable PowerPoint have not yet been imported in this commit; add the verified canonical files from their source location.
