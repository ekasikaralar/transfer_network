import json
import munch
import os
import sys
import argparse

import torch
import numpy as np
from torch import nn
import pandas as pd

import equation as eqn
from solver import DSPL_solver

DATA_ENV = "DEEP_SPLITTING_DATA"
OUT_ENV = "DEEP_SPLITTING_OUT"


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type = str, help = "The path to load json file")
    parser.add_argument('--run_name', type = str, help = 'The name of numerical experiments')
    parser.add_argument('--out_dir', type = str, default = None,
                        help = 'Where to write weights and logs (default: cwd)')

    args = parser.parse_args()

    return args


def load_config(args):
    """Read the config and make every path in it absolute.

    Paths resolve against the directory holding the config file, not the
    working directory, so the solver runs identically from anywhere.  An
    absolute path in the config is left alone, and DEEP_SPLITTING_DATA
    overrides the calibration directory when it is set."""
    path = os.path.abspath(args.config_path)
    base = os.path.dirname(path)

    with open(path) as json_data_file:
        config = json.load(json_data_file)

    def resolve(value):
        return value if os.path.isabs(value) else os.path.normpath(os.path.join(base, value))

    ec = config["eqn_config"]
    # drift_net_dir is absent under the nominal reference policy, where D = 0.
    for key in ("data_dir", "m_net_dir", "drift_net_dir"):
        if ec.get(key) is not None:
            ec[key] = resolve(ec[key])
    if os.environ.get(DATA_ENV):
        ec["data_dir"] = os.path.abspath(os.environ[DATA_ENV])

    return munch.munchify(config)


def output_root(args):
    if args.out_dir:
        return os.path.abspath(args.out_dir)
    return os.path.abspath(os.environ.get(OUT_ENV, "."))


def prepare_directories(args):

    root = output_root(args)
    logs = os.path.join(root, f"logs_{args.run_name}") + os.sep
    weights = os.path.join(root, f"cppweights_{args.run_name}") + os.sep
    os.makedirs(logs, exist_ok = True)
    os.makedirs(weights, exist_ok = True)
    return logs, weights



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

def load_offline(ec):
    """The M and drift networks trained offline.  In separate mode there is one
    of each per grid point, in pooled mode a single one carrying the grid point
    as an input.  The two are chosen independently, so the drift can stay
    separate while M is pooled."""
    _add_offline_modules(ec)

    # One offline network per model grid point, not per splitting step.
    n_model = getattr(ec, "num_model_interval", ec.num_time_interval)
    default_mode = getattr(ec, "offline_mode", "separate")

    def load(net_dir, stem, mode):
        if mode == "separate":
            return [torch.load(os.path.join(net_dir, f"{stem}_{n:03d}.pth"),
                               weights_only = False,
                               map_location = _offline_device())
                    for n in range(n_model)]
        return [torch.load(os.path.join(net_dir, f"{stem}.pth"),
                           weights_only = False,
                           map_location = _offline_device())]

    m_nets = load(ec.m_net_dir, "m_network", getattr(ec, "m_mode", default_mode))
    drift_mode = getattr(ec, "drift_mode", default_mode)
    # The nominal reference policy has D = 0 identically, so there is nothing
    # to load.
    drift_nets = None if drift_mode == "nominal" else load(
        ec.drift_net_dir, "drift_network", drift_mode)
    return m_nets, drift_nets


def main():

    print(torch.__version__)
    args = parse_arguments()

    config = load_config(args)
    save_path_logs, save_path_cppweights = prepare_directories(args)

    # Load the networks that approximate M and D, trained offline
    m_nets, drift_nets = load_offline(config.eqn_config)

    # Initialize equation with the pretrained networks
    bsde = getattr(eqn, config.eqn_config.eqn_name)(
        config.eqn_config, drift_nets = drift_nets, m_nets = m_nets)
    dspl_solver = DSPL_solver(config, bsde)

    training_history = dspl_solver.train(config.net_config.final_iterations,
                                         config.net_config.inner_iterations,
                                         save_path_logs, save_path_cppweights)

    np.savetxt(os.path.join(save_path_logs, "training_history.csv"),
               training_history, delimiter = ',')


if __name__ == '__main__':
    main()
