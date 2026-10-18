# Deep splitting solver

    python3 sample_path_generator.py --config_path configs/main/config.json
    python3 main.py --config_path configs/main/config.json --run_name test_name

The first estimates where the reference process lives at each grid point and
writes `reference_means.csv` and `reference_stds.csv` into the data directory.
The second runs the backward recursion, writing `y_network{n}.pth` and
`z_network{n}.pth` per grid point plus the plain text `z_network{n}_w*.txt`
weights the C++ simulation reads.

Both need the offline networks from `../offline_approximation`, selected by
`offline_mode` in the config, `"separate"` for one network per grid point or
`"pooled"` for one carrying the grid point as an input.

`DEEP_SPLITTING_DATA` overrides the calibration directory and
`DEEP_SPLITTING_OUT` or `--out_dir` overrides where weights and logs go.
