"""Sanity checks of the offline linear program outputs.

M term:  m_term_obj_value_<n>.csv, one optimal value per observation. 
Drift:   d_term_output_<policy>_<n>.csv, one drift vector per observation.
         
    python check_offline.py [--policy value]
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="value")
    args = ap.parse_args()
    od = os.path.join(HERE, "..", "offline_approximation")
    mo = os.path.join(od, "m_term", "separate", "output_data")
    do = os.path.join(od, "drift_term", "separate", "output_data")
    zeta = np.loadtxt(os.path.join(HERE, "data", "zeta_limit.csv"), delimiter=",")
    theta = np.loadtxt(os.path.join(HERE, "data", "theta_hourly.csv"), delimiter=",")

    mfiles = sorted(glob.glob(os.path.join(mo, "m_term_obj_value_*.csv")))
    if mfiles:
        print(f"M term: {len(mfiles)} interval files")
        rows = []
        for f in mfiles:
            n = int(f[-7:-4])
            M = pd.read_csv(f, header=None).to_numpy().ravel()
            x = pd.read_csv(os.path.join(mo, f"state_observations_{n:03d}.csv"), header=None).to_numpy()
            nonneg = (x >= 0).all(axis=1)
            rows.append(dict(n=n, mean=M.mean(), min=M.min(), max=M.max(), frac_zero=(np.abs(M) < 1e-9).mean(),
                             max_on_nonneg=(M[nonneg].max() if nonneg.any() else np.nan),
                             frac_nonneg=nonneg.mean()))
        r = pd.DataFrame(rows)
        print(f"  mean over intervals of mean M {r['mean'].mean():.1f}; min {r['min'].min():.1f}; max {r['max'].max():.1f}")
        print(f"  share of observations with M exactly zero: {r.frac_zero.mean():.3f}")
        print(f"  observations with every x_i >= 0: share {r.frac_nonneg.mean():.3f}, largest M there {r.max_on_nonneg.max():.2e} (must be <= 0)")
        print("  hourly mean M:", np.round(r['mean'].to_numpy()[::12], 1).tolist())

    dfiles = sorted(glob.glob(os.path.join(do, f"d_term_output_{args.policy}_[0-9][0-9][0-9].csv")))
    if dfiles:
        print(f"\ndrift ({args.policy}): {len(dfiles)} interval files")
        rows = []
        for f in dfiles:
            n = int(f[-7:-4])
            D = pd.read_csv(f, header=None).to_numpy()
            x = pd.read_csv(os.path.join(do, f"state_observations_{n:03d}.csv"), header=None).to_numpy()
            drift_total = zeta[n] - theta * x + D
            rows.append(dict(n=n, mean_abs_D=np.abs(D).mean(), max_abs_D=np.abs(D).max(),
                             mean_abs_zeta=np.abs(zeta[n]).mean(), mean_abs_theta_x=np.abs(theta * x).mean(),
                             share_D_zero=(np.abs(D) < 1e-9).mean(), mean_total=drift_total.mean()))
        r = pd.DataFrame(rows)
        print(f"  mean |D| {r.mean_abs_D.mean():.2f} (max {r.max_abs_D.max():.1f}), against mean |zeta| {r.mean_abs_zeta.mean():.2f} and mean |theta x| {r.mean_abs_theta_x.mean():.2f}")
        print(f"  share of zero drift entries {r.share_D_zero.mean():.3f}")
        print("  hourly mean |D|:", np.round(r.mean_abs_D.to_numpy()[::12], 2).tolist())


if __name__ == "__main__":
    main()
