"""Training inputs for the networks that approximate the function D.

D is the drift the reference policy contributes to the state,

    D_i(t, x) = sum_{j in B(i)} (mu_j - theta_b(j)) psitilde_j(t, x)
                - sum_{j in J} P_ji mu_j psitilde_j(t, x),

where psitilde solves the reference linear program over the feasible set
Psi(t, x).  The second sum is the routing feedback: a completion of activity j
sends the caller into class i with probability P_ji, so every activity in the
network contributes to the drift of every class it can route into, not only to
the drift of the class it serves.

The drift vectors themselves are produced by d_term_offline.jl in
output_data.

One network is trained per grid point, so the script writes one state file for
each.  The state box is rebuilt at every interval, since the nominal allocation
psi*(t) and the staffing level both move with t. 
"""

import argparse
import json
import os

import numpy as np

# Environment variable that overrides the calibration directory, so a scheduler
# can point every stage at one location without editing any config.
DATA_ENV = "DEEP_SPLITTING_DATA"


def load_config():
    """Read the config and make every path in it absolute.

    Paths are resolved against the directory holding the config file, not the
    working directory, so the script runs identically from anywhere.  An
    absolute path in the config is left alone, and DEEP_SPLITTING_DATA
    overrides the calibration directory when it is set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.json")
    args = parser.parse_args()

    path = os.path.abspath(args.config_path)
    base = os.path.dirname(path)
    with open(path, "r") as f:
        cfg = json.load(f)

    def resolve(value):
        if value is None:
            return None
        return value if os.path.isabs(value) else os.path.normpath(os.path.join(base, value))

    for key in ("data_dir", "output_dir", "headroom_file"):
        if cfg.get(key) is not None:
            cfg[key] = resolve(cfg[key])
    if os.environ.get(DATA_ENV):
        cfg["data_dir"] = os.path.abspath(os.environ[DATA_ENV])
    return cfg


def nominal_state(xi, agents, scale):
    """x_opt_i(t) = sum_{j in B(i)} psi*_j(t), the nominal number of class i
    callers in service, in units of n.  xi is (num_intervals, I, K) and holds
    the fraction of each pool devoted to each activity, so multiplying by the
    pool headcount and dividing by n gives psi*."""
    return np.einsum("mik,mk->mi", xi, agents / scale)


def state_box(x_opt, headroom, scale):
    """The prelimit number of class i callers is X^n_i = n x_opt_i + sqrt(n) x_i,
    so the centered state x_i runs from -sqrt(n) x_opt_i, where the prelimit
    class is empty, up to the largest centered deviation the class is expected
    to reach."""
    lower = -np.sqrt(scale) * x_opt
    upper = (headroom - scale * x_opt) / np.sqrt(scale)
    return lower, upper


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])

    d = cfg["data_dir"]
    rd = lambda name: np.loadtxt(os.path.join(d, name), delimiter=",")

    I, K, scale = cfg["cust_dim"], cfg["server_dim"], cfg["scale"]
    xi = rd(cfg["xi_file"]).reshape(-1, I, K)
    agents = rd(cfg["agents_file"])

    num_intervals = xi.shape[0]
    x_opt = nominal_state(xi, agents, scale)

    # The headroom is the largest prelimit occupancy a class is expected to
    # reach.  
    if cfg.get("headroom_file"):
        headroom = rd(cfg["headroom_file"])          # (num_intervals, I) or (I,)
        if headroom.ndim == 1:
            headroom = np.tile(headroom, (num_intervals, 1))
    else:
        headroom = np.full_like(x_opt, cfg["headroom"])

    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    n_obs = cfg["num_obs"]

    for n in range(num_intervals):
        lo_x, hi_x = state_box(x_opt[n], headroom[n], scale)
        state = rng.uniform(lo_x, hi_x, size=(n_obs, I))
        np.savetxt(os.path.join(out, f"state_observations_{n:03d}.csv"),
                   state, delimiter=",")

    np.savetxt(os.path.join(out, "x_opt.csv"), x_opt, delimiter=",")
    print(f"{num_intervals} intervals x {n_obs} observations, dimension {I}")
    print(f"  state spans [{(-np.sqrt(scale)*x_opt).min():.2f}, "
          f"{((headroom - scale*x_opt)/np.sqrt(scale)).max():.2f}]")


if __name__ == "__main__":
    main()
