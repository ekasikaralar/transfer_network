"""Test problem: limiting staffing, limiting arrival rates, nominal plan, zeta.

The observed system is the test problem, the nth system of a sequence with
system parameter n.  Its arrival rates lambda^n and staffing levels N are the
estimates from the data on a five minute grid over the horizon.   
Pools are indexed by k, classes by i and activities by j, with s(j) the pool of
activity j.  
R is the input output matrix (service rates net of transfers), A the pool incidence matrix, 
mu_j the service rate and vbar_j the mean value per completion of activity j.

On each time grid point:
    nu_k      = N_k / n,  the limiting staffing level of pool k
    rho_v     = utilization under the value maximizing plan at the observed
                rates: the largest rho at which the plan
                    max vbar mu . psi  s.t.  R psi = lambda^n,  A psi <= rho N,  psi >= 0
                uses every pool fully, A psi* = rho N
    lambda    = lambda^n / (n rho_v),  the limiting arrival rates, eta = 1 / rho_v,
                the smallest factor eta at which the planning problem at
                eta lambda^n / n exhausts every pool
    xi*, psi* = optimal plan of the planning problem at lambda,
                    max tv . xi  s.t.  R D xi = lambda,  A xi <= e,  xi >= 0,
                with D = diag(nu_s(j)) and tv_j = vbar_j mu_j nu_s(j). xi_j is
                the fraction of pool s(j) on activity j and psi* = D xi* the
                nominal number of agents (in units of n) on each activity
    unique    = certificate that xi* is the only optimal plan.  Let d_j be
                the reduced cost of activity j and y_k the dual price of
                pool k. If d_j > 0 for every j with xi*_j = 0 and y_k > 0 for every
                pool, then any other feasible plan either uses an activity
                outside the support of xi* or leaves capacity idle in some
                pool, and loses value either way, so xi* is unique.  This
                is a sufficient condition
    zeta      = (lambda^n - n lambda) / sqrt(n),  the second order term in
                lambda^n = n lambda + sqrt(n) zeta

The heavy traffic assumption, a unique optimal plan with A xi* = e at every
grid point, holds at lambda by construction and is checked.  rho_v is
searched on a grid from above and the largest exhausting grid point is kept.

Writes to calibration/: grid.csv (one row per time grid point with minute, N_total,
eta, rho_v, unique, feasible and exhausting flags at the estimated rates,
n_basic), agents.csv, lambda_n.csv, lambda_limit.csv, zeta_limit.csv,
xi_star.csv, psi_star.csv, basic_activities.csv (rows = grid points).
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import config as C
import network

EXHAUST_TOL, ZERO_TOL, RC_TOL = 1e-6, 1e-7, 1e-7


def solve(c, A_ub, b_ub, A_eq, b_eq):
    """max c . x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  x >= 0; None if infeasible"""
    try:
        r = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=[(0, None)] * len(c), method="highs")
    except ValueError:
        return None
    return r if r.success else None


def value_plan(vmu, R, A, N, lam_n, rho):
    return solve(vmu, A, rho * N, R, lam_n)


def plan(tv, RD, A, lam):
    return solve(tv, A, np.ones(A.shape[0]), RD, lam)


def exhausts(A, x, cap):
    return bool((A @ x >= cap - EXHAUST_TOL * np.maximum(cap, 1.0)).all())


def is_unique(r):
    at_zero = r.x <= ZERO_TOL
    return bool((r.lower.marginals[at_zero] > RC_TOL).all() and (-r.ineqlin.marginals > RC_TOL).all())


def utilization(vmu, R, A, N, lam_n):
    for rho in np.arange(C.RHO_GRID_MAX, C.RHO_GRID_MIN - 1e-12, -C.RHO_GRID_STEP):
        r = value_plan(vmu, R, A, N, lam_n, rho)
        if r is not None and exhausts(A, r.x, rho * N):
            return float(rho)
    return np.nan


def main():
    net = network.build()
    I, J, K = net["I"], net["J"], net["K"]
    activities = pd.read_csv(os.path.join(C.CALIB_DIR, "activities.csv"))
    vmu = activities["vbar"].to_numpy() * net["mu"]
    arr = pd.read_csv(os.path.join(C.RESULTS_DIR, "arrivals_mean_per_5min.csv"), index_col="bin")
    stf = pd.read_csv(os.path.join(C.RESULTS_DIR, "staffing_mean_per_5min.csv"), index_col="bin")
    bins = [b for b in arr.index if C.HORIZON_START <= b * C.STEP < C.HORIZON_END]

    rows, agents, lam_n_all, lam_lim, zeta, xi_all, psi_all, basic = [], [], [], [], [], [], [], []
    for b in bins:
        lam_obs = np.zeros(I)
        lam_obs[:net["nc"]] = arr.loc[b, net["names"]].to_numpy() * 60.0     # per hour, the observed system
        lam_n = C.ARRIVAL_SCALE * lam_obs                                     # the test problem
        N = np.array([max(stf.loc[b, f"Pool {p}"], 0.0) for p in net["pools"]])
        D = np.array([N[net["pool_ix"][p]] / C.N_SCALE for p in net["pool"]])
        RD, tv = net["R"] * D, vmu * D

        # The limiting system
        rho_v = utilization(vmu, net["R"], net["A"], N, lam_obs)
        if np.isnan(rho_v):
            raise RuntimeError(f"no exhausting rho at minute {b * C.STEP}")
        lam = lam_obs / (C.N_SCALE * rho_v) # limiting arrival rates
        r = plan(tv, RD, net["A"], lam)
        if r is None or not exhausts(net["A"], r.x, np.ones(K)):
            raise RuntimeError(f"planning problem at the limiting rates does not exhaust every pool at minute {b * C.STEP}")
        xi = r.x # optimal solution
        r_obs = plan(tv, RD, net["A"], lam_n / C.N_SCALE)
        rows.append(dict(minute=b * C.STEP, N_total=N.sum(), eta=1.0 / rho_v, rho_v=rho_v,
                         unique=is_unique(r),
                         feasible_at_estimated_rates=r_obs is not None,
                         exhausting_at_estimated_rates=exhausts(net["A"], r_obs.x, np.ones(K)) if r_obs is not None else False,
                         n_basic=int((xi > ZERO_TOL).sum())))
        agents.append(N)
        lam_n_all.append(lam_n)
        lam_lim.append(lam)
        zeta.append((lam_n - C.N_SCALE * lam) / np.sqrt(C.N_SCALE))
        xi_all.append(xi)
        psi_all.append(D * xi)
        basic.append((xi > ZERO_TOL).astype(int))
        print(f"  {b * C.STEP // 60:02d}:{b * C.STEP % 60:02d}  rho_v {rho_v:.4f}  eta {1 / rho_v:.4f}", flush=True)

    os.makedirs(C.CALIB_DIR, exist_ok=True)
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(C.CALIB_DIR, "grid.csv"), index=False)
    pd.DataFrame(agents, columns=[f"Pool {p}" for p in net["pools"]]).to_csv(os.path.join(C.CALIB_DIR, "agents.csv"), index=False)
    for fn, M, cols in [("lambda_n.csv", lam_n_all, net["class_names"]),
                        ("lambda_limit.csv", lam_lim, net["class_names"]),
                        ("zeta_limit.csv", zeta, net["class_names"]),
                        ("xi_star.csv", xi_all, list(range(J))),
                        ("psi_star.csv", psi_all, list(range(J))),
                        ("basic_activities.csv", basic, list(range(J)))]:
        pd.DataFrame(np.array(M), columns=cols).to_csv(os.path.join(C.CALIB_DIR, fn), index=False)

    print(f"{len(g)} grid points, average agents {g.N_total.mean():.1f}")
    print(f"rho_v mean {g.rho_v.mean():.4f} [{g.rho_v.min():.4f}, {g.rho_v.max():.4f}], "
          f"eta mean {g.eta.mean():.4f} [{g.eta.min():.4f}, {g.eta.max():.4f}]")
    print(f"unique {int(g.unique.sum())}/{len(g)}, exhausting at estimated rates "
          f"{int(g.exhausting_at_estimated_rates.sum())}, infeasible {int((~g.feasible_at_estimated_rates).sum())}, "
          f"zeta negative at {int((np.array(zeta).sum(axis=1) < 0).sum())}")


if __name__ == "__main__":
    main()
