"""Abandonment rates of the original and rework classes.

A patience observation is one wait in an agent queue.  Original classes use
the first customer subcall, rework classes the second.  A queue time is an
observed patience when the caller abandoned (outcome 11 or 12) and a censored
one otherwise.  Mean patience is the bias corrected Kaplan Meier estimate and
the abandonment rate is 3600 over the mean patience, per hour.

Writes results/abandonment.csv with columns class, kind, n_obs, n_abandoned,
abandon_frac, mean_patience_s, rate_per_hr.
"""
import os
import numpy as np
import pandas as pd
import config as C
from common import add_class_key, drop_abnormal_calls, keep_first_day, raw_files

USECOLS = ["call_id", "cust_subcall", "service", "node", "service_group",
           "segment_start", "queue_time", "service_time", "outcome"]


def km_order(times, flags):
    """Sort by time, observed patiences before censored ones at the same time.
    Queue times are whole seconds, so ties are common, and the estimate depends
    on the order within a tie.  The standard convention is that a caller censored
    at t is still at risk for the abandonments at t; a stable sort makes the
    result the same on every machine."""
    return np.lexsort((-flags, times))


def km_mean(times, flags):
    """Kaplan Meier estimate of the mean. flags = 1 for an observed patience."""
    order = km_order(times, flags)
    t, d = times[order], flags[order]
    n = len(d)
    w = np.zeros(n)
    w[0] = 1 / n
    for i in range(1, n):
        w[i] = w[i - 1] * (n - i + 1) / (n - i) * ((n - i) / (n - i + 1)) ** d[i - 1]
    wd = w * d
    if d[-1] == 0:
        wd[-1] = 1 - wd.sum()
    return float((wd * t).sum())


def km_bias(times, flags):
    order = km_order(times, flags)
    t, d = times[order], flags[order]
    n = len(d)
    b = np.zeros(n - 2)
    b[0] = ((n - 2) / (n - 1)) ** d[0]
    for i in range(1, n - 2):
        b[i] = b[i - 1] * ((n - i - 2) / (n - i - 1)) ** d[i]
    return float(-(n - 1) / n * t[-1] * d[-1] * (1 - d[-2]) * b[-3])


def day_observations(path):
    data = pd.read_csv(path, usecols=USECOLS)
    data = data[(data["queue_time"] < C.QUEUE_TIME_MAX) & (data["service_time"] < C.SERVICE_TIME_MAX)]
    data = drop_abnormal_calls(data)
    data = data[data["cust_subcall"].isin([1, 2])]
    data = data[data["service_group"] == 2] 
    data = (data.sort_values("segment_start").drop_duplicates(["call_id", "cust_subcall"], keep="last"))
    data = data[data["service"].isin(C.SERVICES)]
    data = add_class_key(data)
    data, _ = keep_first_day(data)
    return pd.DataFrame({
        "service": data["service"].to_numpy(), "cnode": data["cnode"].to_numpy(),
        "subcall": data["cust_subcall"].to_numpy(),
        "queue_time": data["queue_time"].to_numpy(),
        "abandon": data["outcome"].isin(C.ABANDON_OUTCOMES).astype(int).to_numpy()})


def main():
    files = raw_files("*_cust_subcalls.csv")
    parts = []
    for i, f in enumerate(files, 1):
        parts.append(day_observations(f))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}", flush=True)
    obs = pd.concat(parts, ignore_index=True)
    rows = []
    for svc, node in C.CLASS_KEYS:
        for sub, kind in [(1, "orig"), (2, "rework")]:
            sel = obs[(obs["service"] == svc) & (obs["cnode"] == node) & (obs["subcall"] == sub)]
            n_obs, n_ab = len(sel), int(sel["abandon"].sum())
            if n_obs >= 10 and n_ab >= 1:
                t, d = sel["queue_time"].to_numpy(float), sel["abandon"].to_numpy(float)
                mean_pat = km_mean(t, d) - km_bias(t, d)
                rate = 3600.0 / mean_pat if mean_pat > 0 else np.nan
            else:
                mean_pat, rate = np.nan, np.nan
            rows.append({"class": C.CLASS_KEY_NAMES[(svc, node)], "kind": kind,
                         "n_obs": n_obs, "n_abandoned": n_ab,
                         "abandon_frac": n_ab / n_obs if n_obs else np.nan,
                         "mean_patience_s": mean_pat, "rate_per_hr": rate})
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(C.RESULTS_DIR, "abandonment.csv"), index=False)
    print("saved abandonment.csv")


if __name__ == "__main__":
    main()
