"""Training inputs for the networks that approximate the function M.

The function M is the optimal value of the linear program

    minimize    sum_j w_j(p) psi_j
    subject to  sum_{j in B(i)} psi_j >= -x_i,          i in I
                sum_{j in S(k)} psi_j >=  0,            k in K
                psi_j <= 0,                             j nonbasic
                psi_j <= sqrt(n) psi*_j(t),             j basic

evaluated at a state x and a gradient p, where

    w_j(p) = (mu_j - theta_b(j)) p_b(j)
             - mu_j sum_i P_ji p_i
             + vbar_j mu_j
             + c_b(j).

This script draws the (x, p) pairs at which that program will be solved.  The
objective values themselves are produced by m_term_offline.jl in output_data.

One network is trained per grid point, so the script writes one state file and
one gradient file for each grid point t_n.  
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
    the fraction
    of each pool devoted to each activity, so multiplying by the pool headcount
    and dividing by n gives psi*."""
    return np.einsum("mik,mk->mi", xi, agents / scale)


def state_box(x_opt, headroom, scale):
    """The prelimit number of class i callers is X^n_i = n x_opt_i + sqrt(n) x_i,
    so the centered state x_i runs from -sqrt(n) x_opt_i, where the prelimit
    class is empty, up to the largest centered deviation the class is expected
    to reach."""
    lower = -np.sqrt(scale) * x_opt
    upper = (headroom - scale * x_opt) / np.sqrt(scale)
    return lower, upper


def gradient_box(cost, theta):
    """Bounds on dV/dx_i, held fixed across the horizon.

    The box is symmetric at half width c_i / theta_i, the holding cost one
    extra class i caller can accrue.  Take the optimal control at x and apply
    it unchanged from x + delta e_i.  It stays feasible, because raising x_i
    only relaxes Yhat_i = x_i + sum_{j in B(i)} psihat_j >= 0 and touches no
    other constraint.  The control is therefore unchanged, so the value term
    sum_j vbar_j mu_j psihat_j does not move and no rework is generated: the
    extra caller simply waits, accruing c_i per unit time and decaying at rate
    theta_i.  Over a remaining horizon tau that costs

        (c_i / theta_i)(1 - exp(-theta_i tau)),

    which increases in tau and so is largest at tau = T.  """
    half = cost / theta
    return -half, half


def main():
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])

    d = cfg["data_dir"]
    rd = lambda name: np.loadtxt(os.path.join(d, name), delimiter=",")

    I, K, scale = cfg["cust_dim"], cfg["server_dim"], cfg["scale"]
    xi = rd(cfg["xi_file"]).reshape(-1, I, K)
    agents = rd(cfg["agents_file"])
    cost = rd(cfg["cost_file"])
    theta = rd(cfg["theta_file"])

    num_intervals = xi.shape[0]
    x_opt = nominal_state(xi, agents, scale)
    lo_p, hi_p = gradient_box(cost, theta)

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
        grad = rng.uniform(lo_p, hi_p, size=(n_obs, I))
        np.savetxt(os.path.join(out, f"state_observations_{n:03d}.csv"),
                   state, delimiter=",")
        np.savetxt(os.path.join(out, f"gradient_observations_{n:03d}.csv"),
                   grad, delimiter=",")

    np.savetxt(os.path.join(out, "x_opt.csv"), x_opt, delimiter=",")
    print(f"{num_intervals} intervals x {n_obs} observations, dimension {I}")
    print(f"  state    spans [{(-np.sqrt(scale)*x_opt).min():.2f}, "
          f"{((headroom - scale*x_opt)/np.sqrt(scale)).max():.2f}]")
    print(f"  gradient spans [{lo_p.min():.3f}, {hi_p.max():.3f}]")


if __name__ == "__main__":
    main()
