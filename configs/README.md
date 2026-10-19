# Experiment configurations

Store versioned configuration files for each validated run. Avoid changing a config in place after recording results; make a new named version and note what changed.

Every config should make these fields explicit: client count/participation, rounds, local epochs, optimizer and learning rate, batch size, dataset partition and alpha, project seed policy, attack client, attack start round, poison rate, trigger definition, target class, method, calibration epochs, weight decay, evaluation populations and output directory.

The current result snapshot is described in docs/EXPERIMENT_PROTOCOL.md. Exact run configs have not yet been imported.
