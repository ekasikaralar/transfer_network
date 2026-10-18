# Dynamic skill based routing with transfers

Everything behind the computational results, from the raw call center data
to the trained policy, in the order it runs.

| Folder | What it holds | Reads | Writes |
|---|---|---|---|
| `raw_data/` | The 84 business days of July to October 2002, three CSV files per day: customer subcalls, agent records, agent events. | | |
| `pipeline/` | Estimates from the raw data (arrivals, staffing, service rates, abandonment, routing) and the calibration of the test problem (values, transfer penalties, limiting rates, nominal plan). | `raw_data/` | `pipeline/results/`, `pipeline/calibration/` |
| `simulation/` | Discrete event simulation of the test problem under the index rules and the trained policy, preemptive with a linear program. | `pipeline/calibration/`, `neural_network/cppweights_<run>/` | `simulation/data/`, `simulation/results/` |
| `inputs/` | The inputs of the deep splitting method and the scripts that keep the configs of the next two folders in step with the calibration. | `pipeline/calibration/`, `simulation/results/` | `inputs/data/` |
| `offline_approximation/` | The offline networks for the function $M$ and the drift $D$ of each reference policy. | `inputs/data/` | `output_data/`, `nn_training/logs_<run>/` |
| `neural_network/` | The deep splitting solver, the backward recursion over the grid, and the trained weight sets `cppweights_<run>/`. | `inputs/data/`, the offline networks | `logs_<run>/`, `cppweights_<run>/` |

    cd pipeline      && ./run_all.sh                       # raw data -> results/ and calibration/
    cd ../simulation && python make_inputs.py && ./build_index.sh && ./run_index.sh 0 500 value fsf
    cd ../inputs     && python make_inputs.py && python make_headroom.py ../simulation/results/value_index.csv.classmax.csv ../simulation/results/fsf_index.csv.classmax.csv
                        ./run_offline.sh                   # inputs, Julia linear programs, offline training
    cd ../neural_network && python sample_path_generator.py --config_path configs/main/config.json
                        python main.py --config_path configs/main/config.json --run_name main
    cd ../simulation && ./build.sh && ./run.sh && ./transfer_sim config_nn.json 0 1000 results/nn.csv

Each folder has its own README with the details.  Every path between folders
is relative, so the project can be moved as a whole.
