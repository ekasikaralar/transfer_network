"""Staffing levels of the nine pools, five minute averages across days.

An agent is on duty between a sign in (event 20 or 21) and the next sign out
(event 30 or 31) of the same agent, minus breaks (events 60, 61, 62).  

Writes results/staffing_mean_per_5min.csv, rows = five minute bins 0..287,
columns = Pool 1..9 and Total, values = mean number of agents on duty.
"""
import os
import numpy as np
import pandas as pd
import config as C
from common import MINUTES_IN_DAY, NBIN5, SECONDS_IN_DAY, raw_files, to_5min

USECOLS = ["agent", "event_start", "event_end", "event_id", "agent_group"]
SIGN_IN, SIGN_OUT, BREAKS = (20, 21), (30, 31), (60, 61, 62)


def second_of_day(epoch):
    return np.asarray(epoch, dtype=np.int64) % SECONDS_IN_DAY


def pool_headcount(data):
    """Per second headcount of one pool on one day."""
    diff = np.zeros(SECONDS_IN_DAY + 1)
    ins = data[data["event_id"].isin(SIGN_IN)]
    outs = data[data["event_id"].isin(SIGN_OUT)]
    for agent, g in ins.groupby("agent"):
        # the agent's sign in and sign out times of the day, in seconds
        t_ins = np.sort(second_of_day(g["event_start"].to_numpy()))
        t_outs = np.sort(second_of_day(outs.loc[outs["agent"] == agent, "event_start"].to_numpy()))
        used = np.zeros(len(t_outs), dtype=bool)
        for t_in in t_ins:
            # pair each sign in with the first later sign out not yet paired,
            # so an agent who signs in and out several times a day gets one
            # on duty interval per pair
            cand = np.where((t_outs >= t_in) & ~used)[0]
            if len(cand):
                used[cand[0]] = True
                t_out = t_outs[cand[0]]
            else:
                # no sign out recorded, the agent is on duty until midnight
                t_out = SECONDS_IN_DAY - 1
            # the headcount rises by one at the sign in and falls by one just
            # after the sign out; the cumulative sum below turns these steps
            # into the number of agents on duty at every second
            diff[t_in] += 1
            diff[t_out + 1] -= 1
    # breaks are subtracted the same way, one agent fewer from the start of
    # the break to its end
    breaks = data[data["event_id"].isin(BREAKS)]
    for s, e in zip(second_of_day(breaks["event_start"]), second_of_day(breaks["event_end"])):
        if e >= s:
            diff[s] -= 1
            diff[e + 1] += 1
    return np.cumsum(diff[:SECONDS_IN_DAY])


def day_profiles(path):
    data = pd.read_csv(path, usecols=USECOLS)
    data = data[data["agent_group"].isin(C.GROUP_TO_POOL)]
    data["pool"] = data["agent_group"].map(C.GROUP_TO_POOL)
    profiles = np.zeros((len(C.POOL_MAP), MINUTES_IN_DAY))
    for pi, pool in enumerate(C.POOL_MAP):
        sub = data[data["pool"] == pool]
        if len(sub):
            profiles[pi] = pool_headcount(sub).reshape(MINUTES_IN_DAY, 60).mean(axis=1)
    return profiles


def main():
    files = raw_files("*_agent_events.csv")
    total5 = np.zeros((len(C.POOL_MAP) + 1, NBIN5))
    for i, f in enumerate(files, 1):
        p = day_profiles(f)
        total5 += to_5min(np.vstack([p, p.sum(axis=0)]))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}", flush=True)
    mean5 = total5 / len(files)
    cols = [f"Pool {p}" for p in C.POOL_MAP] + ["Total"]
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    pd.DataFrame(mean5.T, columns=cols).to_csv(
        os.path.join(C.RESULTS_DIR, "staffing_mean_per_5min.csv"), index_label="bin")
    print(f"saved staffing_mean_per_5min.csv from {len(files)} days")


if __name__ == "__main__":
    main()
