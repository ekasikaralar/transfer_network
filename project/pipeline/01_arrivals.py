"""Arrival rates of the original classes, five minute averages across days.

An arrival of class i is the first customer subcall of a contact whose first
agent service belongs to class i. 

Writes results/arrivals_mean_per_5min.csv, rows = five minute bins 0..287,
columns = class names and Total, values = mean arrivals per minute.
"""
import os
import numpy as np
import pandas as pd
import config as C
from common import (MINUTES_IN_DAY, NBIN5, add_class_key, drop_abnormal_calls,
                    keep_first_day, raw_files, to_5min)

USECOLS = ["call_id", "cust_subcall", "service", "node", "segment_start",
           "queue_time", "service_time", "outcome"]


def day_counts(path):
    data = pd.read_csv(path, usecols=USECOLS)
    data = data[(data["queue_time"] < C.QUEUE_TIME_MAX)
                & (data["service_time"] < C.SERVICE_TIME_MAX)]
    data = drop_abnormal_calls(data)
    data = data[data["cust_subcall"] == 1]
    data = data.sort_values("segment_start").drop_duplicates("call_id", keep="last")
    data = data[data["service"].isin(C.SERVICES)]
    data = add_class_key(data)
    data, dt = keep_first_day(data)
    minute = dt.dt.hour * 60 + dt.dt.minute
    counts = np.zeros((len(C.CLASS_KEYS), MINUTES_IN_DAY))
    for ci, (svc, node) in enumerate(C.CLASS_KEYS):
        m = minute[(data["service"] == svc) & (data["cnode"] == node)]
        np.add.at(counts[ci], m.to_numpy(), 1)
    return counts


def main():
    files = raw_files("*_cust_subcalls.csv")
    total5 = np.zeros((len(C.CLASS_KEYS) + 1, NBIN5))
    for i, f in enumerate(files, 1):
        c = day_counts(f)
        total5 += to_5min(np.vstack([c, c.sum(axis=0)]))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}", flush=True)
    mean5 = total5 / len(files)
    cols = [C.CLASS_KEY_NAMES[k] for k in C.CLASS_KEYS] + ["Total"]
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(mean5.T, columns=cols).to_csv(
        os.path.join(C.RESULTS_DIR, "arrivals_mean_per_5min.csv"), index_label="bin")
    print(f"saved arrivals_mean_per_5min.csv from {len(files)} days")


if __name__ == "__main__":
    main()
