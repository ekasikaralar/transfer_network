# Dynamic Skill-Based Routing for Call Centers with Transfers: A Value-Based Framework

**Authors:** Barış Ata, Ebru Kaşıkaralar

This repository contains the call center data, the code, and the calibrated
inputs behind the computational results of the paper. The code estimates the model primitives
from call center data, calibrates the test problems, trains the offline
networks and the deep splitting solver, and simulates the proposed policy and
the benchmark policies.

---

## Repository structure

The three top-level folders are the three test problems of the paper. The
main test problem holds the complete code. The two variants differ from it
only in their prelimit arrival rates, so their folders hold only the files
that differ and the calibrated inputs of that test problem.

```
.
├── project/                  # main test problem, complete code
│   ├── raw_data/             # the 84 business days of July to October 2002, three CSV files per day
│   ├── pipeline/             # estimation from the raw data and calibration of the test problem
│   │   ├── 01_arrivals.py ... 07_planning.py, common.py, config.py, network.py, run_all.sh
│   │   ├── results/          # five-minute arrival and staffing estimates, service, abandonment and routing tables
│   │   └── calibration/      # limiting rates, second-order terms, nominal plan, activities, grid
│   ├── inputs/               # inputs of the deep splitting method and the scripts that build them
│   │   └── data/             # rates, costs, zeta, sigma, xi, agents, headroom, reference path moments
│   ├── offline_approximation/
│   │   ├── m_term/           # the network for the function M (Julia linear programs, PyTorch training)
│   │   └── drift_term/       # the networks for the drift D of each reference policy
│   ├── neural_network/       # the deep splitting solver
│   │   └── configs/main/config.json   # the configuration of the run reported in the paper
│   └── simulation/           # discrete event simulation of the test problem
│       ├── transfer_sim.cpp  # preemptive simulator with a linear program (all reported results)
│       ├── index_sim.cpp     # non-preemptive simulator, used only to set the headroom of the sampling box
│       ├── config_<rule>.json          # one configuration per benchmark policy
│       ├── config_nn_main.json         # the configuration of the proposed policy
│       └── data/             # the test problem parameters read by the simulators
├── project_arrivals105/      # first variant, 5% more arrivals
│   ├── pipeline/config.py, pipeline/results/, pipeline/calibration/
│   ├── inputs/data/
│   ├── neural_network/configs/main/config.json
│   └── simulation/config_nn_main.json, simulation/data/
└── project_arrivals110/      # second variant, 10% more arrivals, same layout
```

Each folder of `project/` has its own README with the details of that step.
`project/README.md` gives the order in which the steps run.
