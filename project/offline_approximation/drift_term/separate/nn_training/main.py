"""Train one drift network per grid point.

    python main.py --config_path config_value.json --run_name drift_value

Trains every interval by default.  --first and --last restrict the range, so
the intervals can be split across machines or resumed after a stop.
"""

import argparse
import json
import os

import munch
import numpy as np
import torch

from DriftNet import DriftNet


def parse_arguments():
    parser = argparse.ArgumentParser()
    # defaults: the config next to this script, and a run name after it, so
    # that a bare `python main.py` trains the default case
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_value.json'),
                        help="The path to load json file")
    parser.add_argument('--run_name', type=str, default='drift_value', help='The name of numerical experiments')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Where to write logs and weights (default: cwd)')
    parser.add_argument('--first', type=int, default=None, help='First interval to train')
    parser.add_argument('--last', type=int, default=None, help='Last interval, inclusive')
    return parser.parse_args()


# Environment variable that overrides where the trained weights and logs are
# written, so a scheduler can direct output to scratch without editing configs.
OUT_ENV = "DEEP_SPLITTING_OUT"


def load_config(args):
    """Read the config and make every path in it absolute.

    Paths are resolved against the directory holding the config file, not the
    working directory, so training runs identically from anywhere.  An absolute
    path in the config is left alone."""
    path = os.path.abspath(args.config_path)
    base = os.path.dirname(path)
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    obs = config["data_config"]["obs_dir"]
    if not os.path.isabs(obs):
        config["data_config"]["obs_dir"] = os.path.normpath(os.path.join(base, obs))
    return munch.munchify(config)


def output_root(args):
    """Where logs and weights go.  --out_dir wins, then DEEP_SPLITTING_OUT,
    then the working directory."""
    if args.out_dir:
        return os.path.abspath(args.out_dir)
    return os.path.abspath(os.environ.get(OUT_ENV, "."))


def main():
    print(torch.__version__)
    args = parse_arguments()

    config = load_config(args)
    save_path_logs = os.path.join(output_root(args), f"logs_{args.run_name}")
    os.makedirs(save_path_logs, exist_ok=True)

    first = args.first if args.first is not None else 0
    last = args.last if args.last is not None else config.data_config.num_intervals - 1

    for interval in range(first, last + 1):
        model = DriftNet(config, interval)
        training_history = model.train(save_path_logs)
        np.savetxt(os.path.join(save_path_logs, f"training_history_{interval:03d}.csv"),
                   training_history, delimiter=',')


if __name__ == '__main__':
    main()
