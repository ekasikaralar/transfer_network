"""Service rates by pool, class, and layer (original or rework).

A service observation is one agent service from the agent records, attributed
to the serving pool and to the class served. 

Writes results/service_agg_long.csv with columns pool, service, cnode,
cust_subcall, total, count, mean_time, rate_per_hr.
"""
import os

import pandas as pd

import config as C
from common import add_class_key, drop_abnormal_calls, raw_files

USECOLS = ["call_id", "cust_subcall", "agent", "agent_group", "service",
           "node", "seg_type", "outcome", "queue_time", "work_time"]


def day_aggregate(path):
    data = pd.read_csv(path, usecols=USECOLS)
    data = data[data["agent_group"].isin(C.GROUP_TO_POOL)]
    data = drop_abnormal_calls(data)
    data = data[data["cust_subcall"].isin([1, 2])]
    data = data[data["service"].isin(C.SERVICES)]
    data = add_class_key(data)

    consult_ids = data.loc[data["seg_type"].isin([7, 8, 9]), "call_id"].unique()
    is_consult = data["call_id"].isin(consult_ids)
    plain = data[~is_consult]
    dm = data[is_consult].copy()
    dm["work_time"] = dm.groupby(["call_id", "cust_subcall", "agent"])["work_time"].transform("sum")
    merged = dm.drop_duplicates(subset=["call_id", "cust_subcall", "agent"], keep="first") # summing up segment 7,8,9 times
    cols = ["call_id", "cust_subcall", "agent", "agent_group", "service", "cnode",
            "queue_time", "work_time"]
    data = pd.concat([plain[cols], merged[cols]])

    data = data[(data["queue_time"] < C.QUEUE_TIME_MAX)
                & (data["work_time"] < C.SERVICE_TIME_MAX)]
    data["pool"] = data["agent_group"].map(C.GROUP_TO_POOL)
    return (data.groupby(["pool", "service", "cnode", "cust_subcall"])["work_time"]
            .agg(total="sum", count="count").reset_index())


def main():
    files = raw_files("*_agent_records.csv")
    parts = []
    for i, f in enumerate(files, 1):
        parts.append(day_aggregate(f))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}", flush=True)
    agg = (pd.concat(parts).groupby(["pool", "service", "cnode", "cust_subcall"])
           [["total", "count"]].sum().reset_index())
    agg["mean_time"] = agg["total"] / agg["count"]
    agg["rate_per_hr"] = 3600.0 / agg["mean_time"]
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    agg.to_csv(os.path.join(C.RESULTS_DIR, "service_agg_long.csv"), index=False)
    print("saved service_agg_long.csv")


if __name__ == "__main__":
    main()
