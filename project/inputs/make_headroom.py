"""Per class headroom for the state boxes of the offline networks.

The networks are trained on states drawn from a box per class, from the
empty class up to the largest occupancy the class is expected to reach.  One
headroom for every class puts almost all samples of a small class where the
process never goes, so the headroom is taken per class from simulated days
of the reference policy: the largest number of callers of the class in the
system over the replications, times a margin, with a floor.

Reads the .classmax.csv files the index simulator writes, writes
data/headroom.csv (one value per class, callers), and points the two
input_data configs at it.

    python make_headroom.py ../simulation/results/value_index.csv.classmax.csv [more files]
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MARGIN = 1.5     # headroom = MARGIN x the largest occupancy seen
FLOOR = 10       # callers, so that no class gets a degenerate box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("classmax", nargs="+", help=".classmax.csv files from the index simulator")
    args = ap.parse_args()
    peak = None
    for f in args.classmax:
        m = pd.read_csv(f).iloc[:, 1:].to_numpy().max(axis=0)
        peak = m if peak is None else np.maximum(peak, m)
    head = np.maximum(np.ceil(MARGIN * peak), FLOOR)
    np.savetxt(os.path.join(HERE, "data", "headroom.csv"), head, delimiter=",", fmt="%.0f")

    for term in ("m_term", "drift_term"):
        p = os.path.join(HERE, "..", "offline_approximation", term, "separate", "input_data", "config.json")
        cfg = json.load(open(p))
        cfg["headroom_file"] = "../../../../inputs/data/headroom.csv"   # resolved against the config folder
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")

    print("largest occupancy seen per class:", peak.astype(int).tolist())
    print("headroom per class:              ", head.astype(int).tolist())
    print("wrote data/headroom.csv and set headroom_file in both input_data configs")


if __name__ == "__main__":
    main()
