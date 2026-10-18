"""Helpers shared by the estimation scripts."""
import glob
import os

import numpy as np
import pandas as pd

import config as C

MINUTES_IN_DAY = 24 * 60
SECONDS_IN_DAY = 24 * 3600
NBIN5 = MINUTES_IN_DAY // 5

def raw_files(pattern):
    files = sorted(glob.glob(os.path.join(C.RAW_DATA_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {C.RAW_DATA_DIR}")
    return files


def add_class_key(data):
    """Add the node column with node zeroed for every service other than Retail."""
    data = data.copy()
    data["cnode"] = np.where(data["service"] == 1, data["node"], 0)
    return data


def drop_abnormal_calls(data):
    """Remove every record of a call that has an abnormal outcome on any segment."""
    abnormal_calls = data.loc[data["outcome"].isin(C.ABNORMAL_OUTCOMES), "call_id"].unique()
    return data[~data["call_id"].isin(abnormal_calls)]


def keep_first_day(data, ts_col="segment_start"):
    """A daily file may spill a few records into the next day. Keep the first day."""
    dt = pd.to_datetime(data[ts_col], unit="s", utc=True)
    days = dt.dt.date
    uniq = days.unique()
    if len(uniq) > 1:
        keep = days == min(uniq)
        data, dt = data[keep], dt[keep]
    return data, dt


def to_5min(profile_per_min):
    """(rows x 1440) per minute profile -> (rows x 288) five minute averages."""
    return profile_per_min.reshape(profile_per_min.shape[0], NBIN5, 5).mean(axis=2)
