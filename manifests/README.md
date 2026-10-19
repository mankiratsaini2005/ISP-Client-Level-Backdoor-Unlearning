# Run and artifact manifests

Import the run registry and seed manifest from the canonical experiment working tree. Add a SHA-256 manifest for small configuration/source/result files and key artifacts.

Recommended per-run record:
- run ID;
- git commit SHA;
- config path and checksum;
- partition ID / draw;
- explicit RNG seed policy and derived seeds;
- source artifact locations;
- evaluation script and metric denominators;
- runtime boundary;
- output/checkpoint paths;
- whether this run supersedes an earlier record.

Do not invent or backfill unknown values without marking them unknown.
