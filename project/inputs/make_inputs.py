"""Turn the test problem parameters into the inputs of the deep splitting method.

Reads the calibration folder of ../pipeline and writes data/, then sets every
calibration dependent field in the configs of ../offline_approximation and
../neural_network so that both agree with the test problem.  

data/ (M grid points, I classes, K pools):
    xi.csv              M x (I K)  nominal plan xi*, entry cls*K + pool
    agents.csv          M x K      agents on duty per pool, N_k
    lambda.csv          M          total arrival rate of the test problem, per hour
    arr_cdf.csv         M x I      cumulative class distribution of the arrivals
    lambda_limit.csv    M x I      limiting arrival rates lambda
    zeta_limit.csv      M x I      second order term zeta
    sigma_limit.csv     M x (I I)  symmetric square root of Gamma(t)
    mu_hourly.csv       I x K      service rates, zero where there is no activity
    theta_hourly.csv    I          abandonment rates
    cost_rate.csv       I          c_i = h_i + theta_i a_i
    v_activity.csv      I x K      activity values v_j
    vbar_activity.csv   I x K      mean value per completion vbar_j
    kappa_class.csv     I          transfer penalty by destination class
    routing_matrix.csv  (I K) x I  routing probabilities by activity

    python make_inputs.py [--calibration DIR]
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION = os.path.normpath(os.path.join(HERE, "..", "pipeline", "calibration"))
N_SCALE = 100.0
GRID_STEP_MIN = 5
POLICIES = ["fsf", "cmu", "cmu_theta", "myopic", "myopic_theta", "value"]


def covariance_root(lam, mu, psi, P, I, K):
    """sigma(t) with sigma sigma' = Gamma(t) = diag(lambda) + sum_j mu_j psi_j Q_j."""
    S = np.diag(lam.copy())
    for i in range(I):
        for k in range(K):
            if mu[i, k] <= 0 or psi[i, k] == 0:
                continue
            p = P[i * K + k]
            e = np.zeros(I); e[i] = 1.0
            Q = (1.0 - p.sum()) * np.outer(e, e)
            for l in np.nonzero(p)[0]:
                d = np.zeros(I); d[l] = 1.0; d -= e
                Q += p[l] * np.outer(d, d)
            S += mu[i, k] * psi[i, k] * Q
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T, w.min()


def patch(path, updates, drop=()):
    """Update keys of a JSON config in place, nested keys given as 'a.b'."""
    with open(path) as f:
        cfg = json.load(f)
    for key, value in updates.items():
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = value
    for key in drop:
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.get(p, {})
        d.pop(parts[-1], None)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default=os.environ.get("TRANSFER_CALIBRATION", DEFAULT_CALIBRATION))
    args = ap.parse_args()
    cal = args.calibration
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)

    cls = pd.read_csv(os.path.join(cal, "classes.csv"))
    act = pd.read_csv(os.path.join(cal, "activities.csv"))
    pools = pd.read_csv(os.path.join(cal, "pools.csv"))["pool"].tolist()
    P_act = pd.read_csv(os.path.join(cal, "routing_P.csv"), index_col="j").to_numpy()
    grid = pd.read_csv(os.path.join(cal, "grid.csv"))
    agents = pd.read_csv(os.path.join(cal, "agents.csv")).to_numpy()
    lam_n = pd.read_csv(os.path.join(cal, "lambda_n.csv")).to_numpy()
    lam_lim = pd.read_csv(os.path.join(cal, "lambda_limit.csv")).to_numpy()
    zeta = pd.read_csv(os.path.join(cal, "zeta_limit.csv")).to_numpy()
    xi = pd.read_csv(os.path.join(cal, "xi_star.csv")).to_numpy()

    I, K, M = len(cls), len(pools), len(grid)
    pool_ix = {p: k for k, p in enumerate(pools)}
    col = act["cls"].to_numpy() * K + act["pool"].map(pool_ix).to_numpy()

    def by_activity(values):
        m = np.zeros((I, K))
        m[act["cls"], act["pool"].map(pool_ix)] = values
        return m

    mu = by_activity(act["mu"])
    total = lam_n.sum(axis=1)
    cdf = np.cumsum(lam_n / total[:, None], axis=1)
    cdf[:, -1] = 1.0
    xi_wide = np.zeros((M, I * K))
    xi_wide[:, col] = xi
    routing = np.zeros((I * K, I))
    routing[col] = P_act

    sigma = np.zeros((M, I * I))
    min_eig = np.inf
    for m in range(M):
        psi = xi_wide[m].reshape(I, K) * (agents[m] / N_SCALE)[None, :]
        s, w = covariance_root(lam_lim[m], mu, psi, routing, I, K)
        sigma[m] = s.reshape(-1)
        min_eig = min(min_eig, w)

    files = {
        "xi.csv": (xi_wide, "%.8f"), "agents.csv": (agents, "%.4f"), "lambda.csv": (total, "%.6f"),
        "arr_cdf.csv": (cdf, "%.8f"), "lambda_limit.csv": (lam_lim, "%.8f"), "zeta_limit.csv": (zeta, "%.8f"),
        "sigma_limit.csv": (sigma, "%.8f"), "mu_hourly.csv": (mu, "%.6f"),
        "theta_hourly.csv": (cls["theta"].to_numpy(), "%.6f"), "cost_rate.csv": (cls["c"].to_numpy(), "%.6f"),
        "v_activity.csv": (by_activity(act["v"]), "%.6f"),
        "vbar_activity.csv": (by_activity(act["vbar"]), "%.6f"), "kappa_class.csv": (cls["kappa"].to_numpy(), "%.6f"),
        "routing_matrix.csv": (routing, "%.8f"),
    }
    for fn, (arr, fmt) in files.items():
        np.savetxt(os.path.join(out, fn), arr, delimiter=",", fmt=fmt)

    # ---------------------------------------------------------------- configs
    total_time = M * GRID_STEP_MIN / 60.0
    # fallback cap of the state box for every class, 1.5 times the peak staffing
    # as in make_headroom.py; used only until that script writes the per class
    # file and points the configs at it
    headroom = float(math.ceil(1.5 * agents.sum(axis=1).max()))
    dims = {"cust_dim": I, "server_dim": K, "scale": int(N_SCALE)}
    off = os.path.join(HERE, "..", "offline_approximation")

    for term in ("m_term", "drift_term"):
        patch(os.path.join(off, term, "separate", "input_data", "config.json"),
              {**dims, "xi_file": "xi.csv", "headroom": headroom})
        for p in glob.glob(os.path.join(off, term, "separate", "output_data", "config*.json")):
            patch(p, {**dims, "xi_file": "xi.csv"})
        for mode in ("separate", "pooled"):
            for p in glob.glob(os.path.join(off, term, mode, "nn_training", "config*.json")):
                patch(p, {"data_config.num_intervals": M}, drop=("data_config.total_time",))

    for p in glob.glob(os.path.join(HERE, "..", "neural_network", "configs", "*", "config.json")):
        name = os.path.basename(os.path.dirname(p))
        tag = "" if name == "main" else f"{name}_"
        patch(p, {"eqn_config.dim": I, "eqn_config.total_time": total_time,
                  "eqn_config.num_time_interval": M, "eqn_config.num_model_interval": M,
                  "eqn_config.xi_file": "xi.csv",
                  "eqn_config.initial_box": True, "eqn_config.headroom_file": "headroom.csv",
                  "eqn_config.n_scale": N_SCALE,
                  "eqn_config.means_file": f"reference_means_{tag}{M}.csv",
                  "eqn_config.chols_file": f"reference_chols_{tag}{M}.csv"})

    print(f"classes {I}, pools {K}, grid points {M}, horizon {total_time:.1f} h, n {N_SCALE:.0f}")
    print(f"lambda_limit [{lam_lim.min():.4f}, {lam_lim.max():.4f}], zeta [{zeta.min():.4f}, {zeta.max():.4f}]")
    print(f"Gamma min eigenvalue over the grid {min_eig:.3e}; headroom {headroom:.0f}")
    print("wrote data/ and updated the configs of ../offline_approximation and ../neural_network")


if __name__ == "__main__":
    main()
