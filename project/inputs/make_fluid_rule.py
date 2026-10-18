"""The fluid price rule: the policy's weight formula with the planning
problem's duals in place of the gradient of the value function.

At each grid point the planning problem of the pipeline,

    max  tv . xi   s.t.  R D xi = lambda,  A xi <= e,  xi >= 0,

has a dual variable y_i on the flow balance row of every class i, the value
of one more class i arrival per hour, in dollars per caller.  The trained
policy weights activity (i, j) by

    w_ij(p) = c_i + vbar_ij mu_ij + (mu_ij - theta_i) p_i - mu_ij sum_l P_(ij),l p_l

with p the gradient of the cost to go, so the fluid price rule is this
formula with the prices p_i(t) = -y_i(t).  

The simulators run any price table as policy "prices", reading one row per
model grid point and one column per class from prices_path.

Reads  ../pipeline/calibration/ (activities, classes, pools, routing_P,
       agents, lambda_limit) and ../simulation/config_nn.json
Writes ../simulation/data/planning_duals.csv   y,  M x I
       ../simulation/data/fluid_prices.csv     -y, M x I, what the rule uses
       ../simulation/config_fluid.json

    python make_fluid_rule.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "pipeline", "calibration")
SIM = os.path.join(HERE, "..", "simulation")
N_SCALE = 100.0


def planning_duals():
    """y, M x I: the dual of the flow balance row of each class at each grid
    point, d(optimal value)/d(lambda_i), in dollars per caller."""
    act = pd.read_csv(os.path.join(CAL, "activities.csv"))
    cls = pd.read_csv(os.path.join(CAL, "classes.csv"))
    pools = pd.read_csv(os.path.join(CAL, "pools.csv"))["pool"].tolist()
    P = pd.read_csv(os.path.join(CAL, "routing_P.csv"), index_col="j").to_numpy()
    agents = pd.read_csv(os.path.join(CAL, "agents.csv")).to_numpy()
    lam = pd.read_csv(os.path.join(CAL, "lambda_limit.csv")).to_numpy()
    I, J, K, M = len(cls), len(act), len(pools), len(agents)
    pool_ix = {p: k for k, p in enumerate(pools)}

    # input output matrix R (classes x activities) and pool incidence A
    R, A = np.zeros((I, J)), np.zeros((K, J))
    for j, a in act.iterrows():
        R[a.cls, j] += a.mu
        R[:, j] -= P[j] * a.mu                      # completions routed into other classes
        A[pool_ix[a.pool], j] = 1.0
    vmu = (act.vbar * act.mu).to_numpy()
    pool_of = act.pool.map(pool_ix).to_numpy()

    y = np.zeros((M, I))
    for m in range(M):
        D = agents[m][pool_of] / N_SCALE
        r = linprog(-vmu * D, A_ub=A, b_ub=np.ones(K), A_eq=R * D, b_eq=lam[m],
                    bounds=[(0, None)] * J, method="highs")
        if not r.success:
            raise RuntimeError(f"planning problem failed at grid point {m}: {r.message}")
        y[m] = -r.eqlin.marginals       # scipy minimizes -tv.xi, so the sign flips back
    return y


def main():
    y = planning_duals()
    out = os.path.join(SIM, "data")
    np.savetxt(os.path.join(out, "planning_duals.csv"), y, delimiter=",", fmt="%.8f")
    np.savetxt(os.path.join(out, "fluid_prices.csv"), -y, delimiter=",", fmt="%.8f")

    cfg = json.load(open(os.path.join(SIM, "config_nn.json")))
    for key in ("nn_dir", "nn_interval", "num_hidden_layers", "activation", "slope", "n_scale", "nn_scale"):
        cfg.pop(key, None)
    cfg["policy"] = "prices"
    cfg["prices_path"] = "data/fluid_prices.csv"
    with open(os.path.join(SIM, "config_fluid.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"duals: {y.shape[0]} grid points x {y.shape[1]} classes, day average prices -y from {(-y).mean(0).min():.2f} to {(-y).mean(0).max():.2f}")
    print("wrote simulation/data/planning_duals.csv, simulation/data/fluid_prices.csv and simulation/config_fluid.json")


if __name__ == "__main__":
    main()
