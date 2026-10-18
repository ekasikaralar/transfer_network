"""Estimate where the reference process lives at each grid point.

The solver draws its training states at t_n from a fitted distribution rather
than by re-simulating a path every iteration.  This script produces that fit by
simulating the reference process forward,

    Xtilde(t_{n+1}) = Xtilde(t_n)
        + (zeta(t_n) - theta Xtilde(t_n) + Dhat_n(Xtilde(t_n))) dt
        + sigma(t_n) dB(t_n),

from x_0 drawn on the initial box, and recording the mean and standard
deviation of each coordinate at each grid point.

It writes reference_means.csv, dim x num_time_interval, and
reference_chols.csv, one flattened dim x dim Cholesky factor per grid point.
"""

import argparse
import json
import os
import sys

import munch
import numpy as np
import pandas as pd
import torch

import equation as eqn

DATA_ENV = "DEEP_SPLITTING_DATA"

torch.manual_seed(73)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help="The path to load json file")
    parser.add_argument('--num_paths', type=int, default=8192,
                        help='Number of sample paths to simulate')
    parser.add_argument('--ridge', type=float, default=1e-6,
                        help='Ridge added to the covariance before factorizing')
    return parser.parse_args()


def load_config(args):
    path = os.path.abspath(args.config_path)
    base = os.path.dirname(path)
    with open(path) as f:
        config = json.load(f)

    def resolve(v):
        return v if os.path.isabs(v) else os.path.normpath(os.path.join(base, v))

    ec = config["eqn_config"]
    # drift_net_dir is absent under the nominal reference policy, where D = 0.
    for key in ("data_dir", "m_net_dir", "drift_net_dir"):
        if ec.get(key) is not None:
            ec[key] = resolve(ec[key])
    if os.environ.get(DATA_ENV):
        ec["data_dir"] = os.path.abspath(os.environ[DATA_ENV])
    return munch.munchify(config)



def _offline_device():
    """Where to place the offline networks when loading."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _add_offline_modules(ec):
    """Make the offline network classes importable.

    The offline trainers save whole models rather than state dicts, so the
    pickle records the defining module, M_Net or DriftNet.  Those live beside
    the logs directory, so put that directory on the path before loading."""
    for net_dir in (ec.m_net_dir, getattr(ec, "drift_net_dir", None)):
        if net_dir is None:
            continue
        parent = os.path.dirname(os.path.normpath(net_dir))
        if parent not in sys.path:
            sys.path.insert(0, parent)

def load_drift(ec):
    _add_offline_modules(ec)

    n_model = getattr(ec, "num_model_interval", ec.num_time_interval)
    mode = getattr(ec, "drift_mode", getattr(ec, "offline_mode", "separate"))
    if mode == "nominal":
        return None
    if mode == "separate":
        return [torch.load(os.path.join(ec.drift_net_dir, f"drift_network_{n:03d}.pth"),
                           weights_only=False,
                           map_location=_offline_device())
                for n in range(n_model)]
    return [torch.load(os.path.join(ec.drift_net_dir, "drift_network.pth"),
                       weights_only=False,
                           map_location=_offline_device())]


def main():
    args = parse_arguments()
    config = load_config(args)
    ec = config.eqn_config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim, N = ec.dim, ec.num_time_interval
    M = getattr(ec, "num_model_interval", N)      # the coarser model grid
    delta_t = ec.total_time / N
    sqrt_delta_t = np.sqrt(delta_t)
    model_index = lambda n: n * M // N

    bsde = getattr(eqn, ec.eqn_name)(ec, drift_nets=load_drift(ec), m_nets=None)

    d = ec.data_dir
    rd = lambda name: torch.tensor(
        pd.read_csv(os.path.join(d, name), header=None).values,
        dtype=torch.float32).to(device)

    theta = rd(ec.theta_file).reshape(-1)
    zeta = rd(ec.zeta_file)
    sigma = rd(ec.sigma_file).reshape(M, dim, dim)

    S = args.num_paths
    # the same initial distribution as the solver: the per class state box of
    # the first grid point when initial_box is set, else a symmetric box
    u = torch.rand(S, dim, device=device)
    if getattr(ec, "initial_box", False):
        headroom = rd(ec.headroom_file).reshape(-1)
        xi0 = rd(ec.xi_file)[0].reshape(dim, -1)
        agents0 = rd(ec.agents_file)[0]
        x_opt0 = (xi0 * agents0[None, :]).sum(1) / ec.n_scale
        sqrt_n = float(np.sqrt(ec.n_scale))
        lo = -sqrt_n * x_opt0
        hi = (headroom - ec.n_scale * x_opt0) / sqrt_n
        x = lo + u * (hi - lo)
    else:
        x = (u * 2 - 1) * ec.initial_range

    means = torch.zeros(dim, N)
    chols = torch.zeros(N, dim, dim)
    ridge = args.ridge * torch.eye(dim, device=device)

    with torch.no_grad():
        for n in range(N):
            means[:, n] = x.mean(dim=0).cpu()
            chols[n] = torch.linalg.cholesky(
                torch.cov(x.T, correction=0) + ridge).cpu()

            k = model_index(n)
            W = torch.randn(S, dim, device=device) * sqrt_delta_t
            drift = (zeta[k, :] - theta * x + bsde.drift_term(n, x)) * delta_t
            x = x + drift + torch.matmul(W, sigma[k].T)

    # Named by the config, so each splitting grid keeps its own fit.  Gamma_n
    # is a property of the discretized process, so refining num_time_interval
    # means regenerating these.
    np.savetxt(os.path.join(d, ec.means_file), means.numpy(), delimiter=",")
    np.savetxt(os.path.join(d, ec.chols_file),
               chols.reshape(N, dim * dim).numpy(), delimiter=",")

    sd = torch.stack([torch.diagonal(chols[n] @ chols[n].T).sqrt() for n in range(N)])
    print(f"{S} paths over {N} grid points, dimension {dim}")
    print(f"  mean ranges [{means.min():.3f}, {means.max():.3f}]")
    print(f"  sd   ranges [{sd.min():.3f}, {sd.max():.3f}]")
    print(f"  written to {d}/{ec.means_file} and {ec.chols_file}")


if __name__ == '__main__':
    main()
