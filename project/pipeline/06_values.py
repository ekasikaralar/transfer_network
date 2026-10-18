"""Economic primitives: values, transfer penalties, cost rates, abandonment.

    s_o, s_r      resolution probabilities for original and rework classes
                  s_o (1 - p_rework) + s_r p_rework = SBAR,  s_r = FCR_RATIO s_o
    u_i           customer value   DELTA_RHO m_i / (1 + DELTA - RHO)
    gamma_j       matching factor by the position of the pool in the share order
    v_j           activity value   s_j * u_b(j) * gamma_j
    gammabar_l    share weighted average of gamma over the activities of rework class l
    kappa_l       transfer penalty KAPPA0 * s_r * u_l * gammabar_l
    vbar_j        mean value per completion  q_j * v_j - sum_l P_jl * kappa_l
    theta_i       abandonment rate, a rework class with few abandonments takes
                  the rate of its original class
    h_i, a_i, c_i holding cost, abandonment penalty, c_i = h_i + theta_i a_i

Writes calibration/classes.csv, calibration/activities.csv,
calibration/routing_P.csv, calibration/pools.csv.
"""
import os

import numpy as np
import pandas as pd

import config as C
import network


def main():
    net = network.build()
    I, J, nc = net["I"], net["J"], net["nc"]
    names = net["names"]

    # resolution probabilities
    rt = pd.read_csv(os.path.join(C.RESULTS_DIR, "routing_class_level.csv"))
    rt = rt[rt["origin"].isin(names)]
    p_rework = float((rt["n"] * rt["rework_pct"] / 100.0).sum() / rt["n"].sum())
    s_o = C.SBAR / ((1.0 - p_rework) + C.FCR_RATIO * p_rework)
    s_r = C.FCR_RATIO * s_o

    # customer values, the rework class has the same value as its original class
    u = np.array([C.DELTA_RHO * C.MARGINS[n] / (1.0 + C.DELTA - C.RHO) for n in names] * 2)

    # matching factors by position in the share order within each class
    gamma, position, share = np.zeros(J), np.zeros(J, dtype=int), np.zeros(J)
    for c in range(I):
        js = np.where(net["cls"] == c)[0]
        order = js[np.argsort(-net["nobs"][js], kind="stable")]
        for r, j in enumerate(order):
            position[j] = r + 1
            gamma[j] = C.GAMMA_LADDER[min(r, len(C.GAMMA_LADDER) - 1)]
            share[j] = net["nobs"][j] / net["nobs"][js].sum()

    # activity values and transfer penalties
    s_act = np.where(net["layer"] == 1, s_o, s_r)
    v = s_act * u[net["cls"]] * gamma
    gammabar = np.ones(I)
    for c in range(nc, I):
        js = np.where(net["cls"] == c)[0]
        gammabar[c] = float((share[js] * gamma[js]).sum())
    kappa = np.zeros(I)
    kappa[nc:] = C.KAPPA0 * s_r * u[nc:] * gammabar[nc:]
    q = 1.0 - net["P"].sum(axis=1)
    vbar = np.where(net["layer"] == 1, q * v - net["P"] @ kappa, v)

    # abandonment rates and cost rates
    ab = pd.read_csv(os.path.join(C.RESULTS_DIR, "abandonment.csv"))
    theta = np.full(I, np.nan)
    for lay, kind in [(1, "orig"), (2, "rework")]:
        sub = ab[ab["kind"] == kind].set_index("class")
        for i, n in enumerate(names):
            idx = i if lay == 1 else nc + i
            if n in sub.index:
                r, nab = float(sub.loc[n, "rate_per_hr"]), float(sub.loc[n, "n_abandoned"])
                if np.isfinite(r) and r > 0 and nab >= C.MIN_ABANDONMENTS:
                    theta[idx] = r
    for i in range(nc):
        if not np.isfinite(theta[nc + i]):
            theta[nc + i] = theta[i]
    assert np.isfinite(theta).all(), "missing abandonment rate for an original class"
    h = np.array([C.HOLDING_COST[n] for n in names] * 2, dtype=float)
    a = h / C.CALLS_PER_HOUR
    c_rate = h + theta * a

    os.makedirs(C.CALIB_DIR, exist_ok=True)
    pd.DataFrame(dict(cls=range(I), name=net["class_names"], layer=[1] * nc + [2] * nc,
                      m=[C.MARGINS[n] for n in names] * 2, u=u, gammabar=gammabar, kappa=kappa,
                      theta=theta, h=h, a=a, c=c_rate)
                 ).to_csv(os.path.join(C.CALIB_DIR, "classes.csv"), index=False)
    pd.DataFrame(dict(j=range(J), cls=net["cls"], name=[net["class_names"][c] for c in net["cls"]],
                      layer=net["layer"], pool=net["pool"], mu=net["mu"], nobs=net["nobs"],
                      share=share, position=position, gamma=gamma, s=s_act, v=v, q=q, vbar=vbar)
                 ).to_csv(os.path.join(C.CALIB_DIR, "activities.csv"), index=False)
    pd.DataFrame(net["P"], columns=net["class_names"]).to_csv(
        os.path.join(C.CALIB_DIR, "routing_P.csv"), index_label="j")
    pd.DataFrame(dict(pool=net["pools"], agent_groups=[" ".join(map(str, C.POOL_MAP[p])) for p in net["pools"]])
                 ).to_csv(os.path.join(C.CALIB_DIR, "pools.csv"), index=False)

    print(f"rework share {p_rework:.4f}, s_o {s_o:.4f}, s_r {s_r:.4f}")
    print(f"v_j originals [{v[net['layer']==1].min():.2f}, {v[net['layer']==1].max():.2f}], "
          f"rework [{v[net['layer']==2].min():.2f}, {v[net['layer']==2].max():.2f}], "
          f"kappa [{kappa[nc:].min():.2f}, {kappa[nc:].max():.2f}]")
    print(f"{J} activities, {I} classes, {net['K']} pools. saved classes.csv, activities.csv, routing_P.csv, pools.csv")


if __name__ == "__main__":
    main()
