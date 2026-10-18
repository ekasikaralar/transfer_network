# Inputs of the deep splitting method

Turns the test problem of `../pipeline/calibration` into the files that
`../offline_approximation` and `../neural_network` read, and keeps their
configs in step with the calibration.

    python make_inputs.py        # ../pipeline/calibration -> data/, and sets the configs
    python make_headroom.py ../simulation/results/value_index.csv.classmax.csv \
                            ../simulation/results/fsf_index.csv.classmax.csv   # per class state boxes
    ./run_offline.sh             # offline stage in ../offline_approximation: inputs (numpy), LPs (Julia), training (torch)
    python check_offline.py      # sanity checks of the linear program outputs
    cd ../neural_network
    python sample_path_generator.py --config_path configs/main/config.json
    python main.py --config_path configs/main/config.json --run_name main
    cd ../simulation
    ./transfer_sim config_nn.json 0 1000 results/nn.csv      # or ./index_sim

`make_inputs.py` writes the fourteen files of `data/` and updates every field
of the configs that depends on the calibration, the dimensions, the grid, the
horizon, the file names and the headroom of the state box.  

`make_headroom.py` sizes the state box of each class from the largest occupancy
the class reaches in simulated days of the index rules, 1.5 times that with a
floor of 10 callers, writes `data/headroom.csv` and points the two input configs
of the offline stage at it.  

`make_fluid_rule.py` builds the fluid price rule, the policy's weight formula
with the planning problem's class duals in place of the gradient.  It solves
the planning problem at every grid point and writes the duals and the prices
`-y` to `../simulation/data/`, with `../simulation/config_fluid.json` running
them as policy `prices`.

`check_offline.py` summarizes the linear program outputs, the sign and size of
$M$ and the size of the drift against $\zeta$ and $\theta x$.

## Who reads what

| Folder | What it does | Reads | Writes |
|---|---|---|---|
| `../offline_approximation/` | Networks for the function $M$ and for the drift $D$ of each reference policy. Inputs are drawn by `*_input_data.py`, the linear programs are solved by `*_offline.jl`, the networks are trained by `nn_training/main.py`. | `data/` | observations in `output_data/`, weights in `nn_training/logs_<run>` |
| `../neural_network/` | The backward recursion of the deep splitting method. `sample_path_generator.py` first estimates where the reference process lives at each grid point. | `data/`, the offline networks | `logs_<run>` with `.pth` weights, `cppweights_<run>` with the text weights the simulation reads |
| `../simulation/` | Runs the trained policy as policy `nn` of both simulators, reading `../neural_network/cppweights_<run>`. | `data/` of its own, the text weights | `results/` |

## Requirements

Python with numpy and pandas for `make_inputs.py` and the input scripts, torch
and munch for the training, Julia with JuMP, HiGHS, CSV, DataFrames and JSON
for the linear programs.  