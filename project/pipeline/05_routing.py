"""Routing probabilities from the first agent service to the second.

Within a subcall, the segment with the longest
service time defines the service delivered and the serving agent group.

Writes results/routing_class_level.csv, one row per original class, and
results/routing_class_pool_level.csv, one row per (original class, pool of
the first service), contacts whose first agent group has no pool left out.
Columns: origin, n, rework_pct, exit_pct, and the percentage routed to each
rework class.
"""
import csv
import os
from collections import Counter, defaultdict

import config as C
from common import raw_files


def class_key(service, node):
    return (service, node if service == 1 else 0)

def process_file(path, n_k, n_kl, n_kg, n_kgl):
    with open(path) as f:
        r = csv.reader(f)
        idx = {h: i for i, h in enumerate(next(r))}
        ci, cs, sg, sv = idx["call_id"], idx["cust_subcall"], idx["service_group"], idx["service"]
        pa, st, ag, nd = idx["party_answered"], idx["service_time"], idx["agent_group"], idx["node"]
        calls = defaultdict(dict)
        for row in r:
            if row[sg] != "2" or int(row[pa]) <= 10000:
                continue
            sub, stv = int(row[cs]), int(row[st])
            cur = calls[row[ci]].get(sub)
            if cur is None or stv > cur[3]:
                calls[row[ci]][sub] = (int(row[sv]), int(row[nd]), int(row[ag]), stv)
    for subs in calls.values():
        order = sorted(subs)
        first = subs[order[0]]
        if first[0] not in C.SERVICES:
            continue
        k, pool = class_key(first[0], first[1]), C.GROUP_TO_POOL.get(first[2])
        second = subs[order[1]] if len(order) > 1 else None
        l = class_key(second[0], second[1]) if second is not None and second[0] in C.SERVICES else None
        n_k[k] += 1
        n_kg[(k, pool)] += 1
        if l is not None:
            n_kl[(k, l)] += 1
            n_kgl[((k, pool), l)] += 1
    return len(calls)


def write_table(fname, index_rows, totals, pairs):
    with open(os.path.join(C.RESULTS_DIR, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["origin", "n", "rework_pct", "exit_pct"] + [C.CLASS_KEY_NAMES[k] for k in C.CLASS_KEYS])
        for label, key in index_rows:
            n = totals.get(key, 0)
            if n == 0:
                continue
            counts = [pairs.get((key, l), 0) for l in C.CLASS_KEYS]
            rework = sum(counts)
            w.writerow([label, n, f"{rework / n * 100:.2f}", f"{(1 - rework / n) * 100:.2f}"]
                       + [f"{c / n * 100:.4f}" for c in counts])


def main():
    files = raw_files("*_cust_subcalls.csv")
    n_k, n_kl, n_kg, n_kgl = Counter(), Counter(), Counter(), Counter()
    for i, path in enumerate(files, 1):
        nc = process_file(path, n_k, n_kl, n_kg, n_kgl)
        print(f"[{i}/{len(files)}] {os.path.basename(path)} calls={nc}", flush=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    write_table("routing_class_level.csv", [(C.CLASS_KEY_NAMES[k], k) for k in C.CLASS_KEYS], n_k, n_kl)
    keys = sorted((kg for kg in n_kg if kg[1] is not None), key=lambda kg: (kg[0], kg[1]))
    rows = [(f"{C.CLASS_KEY_NAMES[k]} | pool {g}", (k, g)) for k, g in keys]
    write_table("routing_class_pool_level.csv", rows, n_kg, n_kgl)
    tot, rew = sum(n_k.values()), sum(n_kl.values())
    print(f"contacts {tot}, with a second service {rew} ({rew / tot * 100:.2f}%)")
    print("saved routing_class_level.csv and routing_class_pool_level.csv")


if __name__ == "__main__":
    main()
