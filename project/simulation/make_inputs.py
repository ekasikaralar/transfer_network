"""Turn the test problem parameters into the files the simulator reads.

Reads the calibration folder of the data pipeline and writes data/ plus one
config_<policy>.json per index rule.  

    lambda.csv          M          total arrival rate per interval, per hour
    arr_cdf.csv         M x I      cumulative class distribution of the arrivals
    agents.csv          M x K      agents on duty per pool
    xi.csv              M x (I K)  nominal plan, entry cls*K + pool
    mu_hourly.csv       I x K      service rates, zero where there is no activity
    theta_hourly.csv    I          abandonment rates
    cost_rate.csv       I          c_i = h_i + theta_i a_i, used by the index rules
    holding_cost.csv    I          h_i, per caller in queue per hour
    abandon_penalty.csv I          a_i, per abandonment
    v_activity.csv      I x K      activity values v_j
    vbar_activity.csv   I x K      mean value per completion vbar_j
    kappa_class.csv     I          transfer penalty kappa_l by destination class
    routing_matrix.csv  (I K) x I  routing probabilities by activity

    python make_inputs.py [--calibration DIR] [--iterations N] [--nn-dir DIR]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION = os.path.normpath(os.path.join(HERE, "..", "pipeline", "calibration"))
POLICIES = ["cmu", "cmu_theta", "fsf", "myopic", "myopic_theta", "value"]
GRID_STEP_MIN = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=os.environ.get("TRANSFER_CALIBRATION", DEFAULT_CALIBRATION))
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--nn-dir", default="../neural_network/cppweights_main",
                    help="folder with the z_network<n>_w*.txt weights the solver writes")
    args = ap.parse_args()
    cal = args.calibration
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)

    cls = pd.read_csv(os.path.join(cal, "classes.csv"))
    act = pd.read_csv(os.path.join(cal, "activities.csv"))
    pools = pd.read_csv(os.path.join(cal, "pools.csv"))["pool"].tolist()
    P = pd.read_csv(os.path.join(cal, "routing_P.csv"), index_col="j").to_numpy()
    grid = pd.read_csv(os.path.join(cal, "grid.csv"))
    agents = pd.read_csv(os.path.join(cal, "agents.csv")).to_numpy()
    lam_n = pd.read_csv(os.path.join(cal, "lambda_n.csv")).to_numpy()
    xi = pd.read_csv(os.path.join(cal, "xi_star.csv")).to_numpy()

    I, K, M, J = len(cls), len(pools), len(grid), len(act)
    pool_ix = {p: k for k, p in enumerate(pools)}
    col = act["cls"].to_numpy() * K + act["pool"].map(pool_ix).to_numpy()

    def by_activity(values):
        m = np.zeros((I, K))
        m[act["cls"], act["pool"].map(pool_ix)] = values
        return m

    total = lam_n.sum(axis=1)
    cdf = np.cumsum(lam_n / total[:, None], axis=1)
    cdf[:, -1] = 1.0
    xi_wide = np.zeros((M, I * K))
    xi_wide[:, col] = xi
    routing = np.zeros((I * K, I))
    routing[col] = P

    files = {
        "lambda.csv": (total, "%.6f"), "arr_cdf.csv": (cdf, "%.8f"), "agents.csv": (agents, "%.4f"),
        "xi.csv": (xi_wide, "%.8f"), "mu_hourly.csv": (by_activity(act["mu"]), "%.6f"),
        "theta_hourly.csv": (cls["theta"].to_numpy(), "%.6f"), "cost_rate.csv": (cls["c"].to_numpy(), "%.6f"),
        "holding_cost.csv": (cls["h"].to_numpy(), "%.6f"), "abandon_penalty.csv": (cls["a"].to_numpy(), "%.6f"),
        "v_activity.csv": (by_activity(act["v"]), "%.6f"), "vbar_activity.csv": (by_activity(act["vbar"]), "%.6f"),
        "kappa_class.csv": (cls["kappa"].to_numpy(), "%.6f"), "routing_matrix.csv": (routing, "%.8f"),
    }
    for fn, (arr, fmt) in files.items():
        np.savetxt(os.path.join(out, fn), arr, delimiter=",", fmt=fmt)

    for policy in POLICIES:
        cfg = dict(class_no=I, station_no=K, num_interval=M, num_iterations=args.iterations,
                   grid_step_min=GRID_STEP_MIN, policy=policy,
                   lambda_path="data/lambda.csv", arr_cdf_path="data/arr_cdf.csv",
                   agents_path="data/agents.csv", xi_path="data/xi.csv",
                   mu_hourly_path="data/mu_hourly.csv", theta_hourly_path="data/theta_hourly.csv",
                   cost_rate_path="data/cost_rate.csv", holding_cost_path="data/holding_cost.csv",
                   abandon_penalty_path="data/abandon_penalty.csv",
                   value_path="data/v_activity.csv", vbar_path="data/vbar_activity.csv",
                   kappa_path="data/kappa_class.csv", routing_path="data/routing_matrix.csv")
        with open(os.path.join(HERE, f"config_{policy}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")

    # the trained policy: same inputs, plus the gradient networks of the solver.
    # The solver may use a finer splitting grid than the model grid, so the
    # number of networks is read from the weight folder when it exists.
    nn_dir = os.path.join(HERE, args.nn_dir) if not os.path.isabs(args.nn_dir) else args.nn_dir
    n_nets = len([f for f in os.listdir(nn_dir) if f.startswith("z_network") and f.endswith("_w0.txt")]) if os.path.isdir(nn_dir) else 0
    if n_nets == 0:
        print(f"no networks found in {args.nn_dir}; config_nn.json assumes one per model interval")
    cfg = dict(class_no=I, station_no=K, num_interval=M, num_iterations=args.iterations,
               grid_step_min=GRID_STEP_MIN, policy="nn",
               nn_dir=args.nn_dir, nn_interval=n_nets or M, num_hidden_layers=4, activation="eLU", slope=0.0,
               n_scale=100.0,
               lambda_path="data/lambda.csv", arr_cdf_path="data/arr_cdf.csv",
               agents_path="data/agents.csv", xi_path="data/xi.csv",
               mu_hourly_path="data/mu_hourly.csv", theta_hourly_path="data/theta_hourly.csv",
               cost_rate_path="data/cost_rate.csv", holding_cost_path="data/holding_cost.csv",
               abandon_penalty_path="data/abandon_penalty.csv",
               value_path="data/v_activity.csv", vbar_path="data/vbar_activity.csv",
               kappa_path="data/kappa_class.csv", routing_path="data/routing_matrix.csv")
    with open(os.path.join(HERE, "config_nn.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"classes {I}, pools {K}, activities {J}, intervals {M}")
    print(f"arrival rate per hour {total.min():.0f} to {total.max():.0f}; wrote data/ and {len(POLICIES)} configs")


if __name__ == "__main__":
    main()
