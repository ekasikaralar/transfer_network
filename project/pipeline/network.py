"""The activity network from the estimates in results/.

An activity is a (pool, class, layer) cell of service_agg_long.csv that
passes the activity filter in config.  Activities are ordered by layer, pool,
class.  The routing matrix P (activities x classes) comes from the pool
conditioned routing table, R is the input output matrix, A the capacity
consumption matrix.
"""
import os
import numpy as np
import pandas as pd
import config as C


def build():
    agg = pd.read_csv(os.path.join(C.RESULTS_DIR, "service_agg_long.csv"))
    agg = agg[agg.cust_subcall.isin([1, 2])]
    pool_layer_total = agg.groupby(["pool", "cust_subcall"])["count"].transform("sum")
    agg = agg[(agg["count"] >= C.MIN_CELL) & (agg["count"] >= C.ELIM_SHARE * pool_layer_total)]

    keys, pools = list(C.CLASS_KEYS), list(C.POOL_MAP)
    nc = len(keys)
    cls_ix = {(k, 1): i for i, k in enumerate(keys)}
    cls_ix.update({(k, 2): nc + i for i, k in enumerate(keys)})
    pool_ix = {p: i for i, p in enumerate(pools)}

    activities = []
    for _, r in agg.iterrows():
        k = (int(r.service), int(r.cnode) if r.service == 1 else 0)
        activities.append(dict(pool=int(r["pool"]), key=k, layer=int(r.cust_subcall),
                         cls=cls_ix[(k, int(r.cust_subcall))],
                         mu=float(r.rate_per_hr), nobs=int(r["count"])))
    activities.sort(key=lambda a: (a["layer"], a["pool"], a["cls"]))
    I, K, J = 2 * nc, len(pools), len(activities) # number of classes, number of pools, number of activities

    rt = pd.read_csv(os.path.join(C.RESULTS_DIR, "routing_class_pool_level.csv")).set_index("origin")
    P = np.zeros((J, I)) # routing matrix
    for j, a in enumerate(activities):
        if a["layer"] != 1:
            continue
        lab = f"{C.CLASS_KEY_NAMES[a['key']]} | pool {a['pool']}"
        if lab in rt.index:
            for l, kk in enumerate(keys):
                P[j, nc + l] = float(rt.loc[lab, C.CLASS_KEY_NAMES[kk]]) / 100.0

    R = np.zeros((I, J))
    A = np.zeros((K, J))
    for j, a in enumerate(activities):
        R[a["cls"], j] += a["mu"]
        for i in range(I):
            if i != a["cls"]:
                R[i, j] -= P[j, i] * a["mu"]
        A[pool_ix[a["pool"]], j] = 1.0

    names = [C.CLASS_KEY_NAMES[k] for k in keys]
    return dict(activities=activities, R=R, A=A, P=P, I=I, K=K, J=J, nc=nc, pools=pools, pool_ix=pool_ix,
                names=names, class_names=names + [f"{n} [rework]" for n in names],
                mu=np.array([a["mu"] for a in activities]), nobs=np.array([a["nobs"] for a in activities]),
                layer=np.array([a["layer"] for a in activities]), pool=np.array([a["pool"] for a in activities]),
                cls=np.array([a["cls"] for a in activities]))
