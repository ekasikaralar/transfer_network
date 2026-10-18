# Pipeline

    ./run_all.sh          # or: PYTHON=/path/to/python ./run_all.sh
    ./run_all.sh 6 7      # only the calibration steps

Needs Python with numpy, pandas and scipy. 
The raw data folder is `../raw_data` by default, set in
`config.py` or by the environment variable `TRANSFER_RAW_DATA`. Every other
setting, classes, pools, filters, calibration parameters, horizon, is in
`config.py`.

| Step | Reads | Writes to `results/` |
|---|---|---|
| `01_arrivals.py` | `*_cust_subcalls.csv` | `arrivals_mean_per_5min.csv` |
| `02_staffing.py` | `*_agent_events.csv` | `staffing_mean_per_5min.csv` |
| `03_service_rates.py` | `*_agent_records.csv` | `service_agg_long.csv` |
| `04_abandonment.py` | `*_cust_subcalls.csv` | `abandonment.csv` |
| `05_routing.py` | `*_cust_subcalls.csv` | `routing_class_level.csv`, `routing_class_pool_level.csv` |

| Step | Reads | Writes to `calibration/` |
|---|---|---|
| `06_values.py` | `results/` | `classes.csv` (u, gammabar, kappa, theta, h, a, c per class), `activities.csv` (pool, class, mu, gamma, v, vbar per activity), `routing_P.csv`, `pools.csv` |
| `07_planning.py` | `results/`, `calibration/activities.csv` | `grid.csv` (eta, rho_v, uniqueness per grid point), `agents.csv`, `lambda_n.csv`, `lambda_limit.csv`, `zeta_limit.csv`, `xi_star.csv`, `psi_star.csv`, `basic_activities.csv` |

`network.py` builds the activity network from `results/` and is used by steps
6 and 7. 
