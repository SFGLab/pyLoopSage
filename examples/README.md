# Example configuration files

Each file changes only the settings needed for one use case; everything else keeps its default.
Edit the file paths, then run `loopsage -c examples/<file>.ini`. All keys are listed in `config_example.ini`.

| File | Use case |
|---|---|
| `01_loops_only.ini` | Loops from a BEDPE file, Monte Carlo only |
| `02_chiapet_motif_orientation.ini` | ChIA-PET loops, PET filter and CTCF motif orientation |
| `03_with_3d_structures.ini` | Add 3D molecular dynamics, viewer and videos |
| `04_annealing.ini` | Simulated annealing (temperature decays) |
| `05_compartments_epigenetic.ini` | A/B compartments and the epigenetic model |
| `06_chip_tracks.ini` | BigWig tracks as binding energy and LEF loading |
| `07_two_lef_families.ini` | Two LEF families with different folding strength |
| `08_cluster_replicas.ini` | Long production run with replicas and a GPU |
| `09_quick_test.ini` | Small test with the shipped demo data |
| `10_optimize_parameters.ini` | Fit K, f and b first (`--optimize`), then run with them |
