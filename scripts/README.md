# Scripts

Import the actual experiment launchers and evaluation/plotting scripts from the canonical working tree.

When adding a script, document:
- the command-line invocation;
- required config and input files;
- output paths;
- expected runtime and hardware assumptions;
- random seed/partition policy;
- the run ID and metrics it generates.

Keep scripts for analysis/replotting separate from expensive training launchers. Never imply a script was executed unless the run and output are recorded.
