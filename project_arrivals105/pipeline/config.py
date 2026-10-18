"""All settings of the pipeline in one place."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ paths
RAW_DATA_DIR = os.environ.get(
    "TRANSFER_RAW_DATA",
    os.path.join(HERE, "..", "raw_data"))
RESULTS_DIR = os.path.join(HERE, "results")          # estimates from the raw data
CALIB_DIR = os.path.join(HERE, "calibration")        # values and planning outputs

# ------------------------------------------------------------------ classes and pools
SERVICE_NAMES = {
    1: "Retail", 2: "Premier", 3: "Business", 4: "Platinum",
    5: "Consumer Loans", 6: "Online Banking", 7: "EBO", 8: "Telesales",
    9: "Subanco", 10: "Case Quality", 11: "Priority Service",
}
SERVICES = list(SERVICE_NAMES)
# Retail is split into three classes by node.  A class key is (service, node),
# with node = 0 for every service other than Retail.
CLASS_KEYS = [(1, 1), (1, 2), (1, 3)] + [(s, 0) for s in SERVICES if s != 1]
CLASS_KEY_NAMES = {
    (1, 1): "Retail (node 1)", (1, 2): "Retail (node 2)", (1, 3): "Retail (node 3)",
    **{(s, 0): SERVICE_NAMES[s] for s in SERVICES if s != 1},
}
# Merged server pools, by agent group code.  15/16 and 19/20 are the same
# group whose code changes across days.
POOL_MAP = {1: [1], 2: [5], 3: [9], 4: [26, 28, 30], 5: [19, 20],
            6: [31], 7: [33], 8: [34], 9: [15, 16, 38, 39, 40]}
GROUP_TO_POOL = {g: p for p, groups in POOL_MAP.items() for g in groups}

# ------------------------------------------------------------------ data filters
ABNORMAL_OUTCOMES = [4, 13, 14, 23, 30, 40, 50]
QUEUE_TIME_MAX = 900        # seconds
SERVICE_TIME_MAX = 1800     # seconds
ABANDON_OUTCOMES = [11, 12]
MIN_ABANDONMENTS = 10       # a rework class with fewer takes the original class abandonment rate

# To remove outliers due to random assignments
# Activity filter: a (pool, class, layer) cell is an activity when it has at
# least MIN_CELL observations and at least ELIM_SHARE of the pool's volume in
# that layer.
MIN_CELL = 100
ELIM_SHARE = 0.001

# ------------------------------------------------------------------ value calibration
SBAR = 0.69              # average first call resolution
FCR_RATIO = 0.86         # resolution is 14% lower after a transfer
RHO = 0.90               # annual retention probability
DELTA = 0.12             # annual discount rate
DELTA_RHO = 0.0139       # retention lift (1 - RHO) (1/3) / 2.4, rounded to 1.39%
KAPPA0 = 0.12            # value after a transfer is 12% lower
GAMMA_LADDER = [1.0, 0.9, 0.8, 0.7]   # matching factor by position in the share order
MARGINS = {              # annual margin m_i in dollars by service type
    "Retail (node 1)": 75, "Retail (node 2)": 75, "Retail (node 3)": 75,
    "Premier": 125, "Business": 150, "Platinum": 200,
    "Consumer Loans": 65, "Online Banking": 65, "Telesales": 65,
    "EBO": 55, "Subanco": 55, "Case Quality": 55, "Priority Service": 200,
}
HOLDING_COST = {         # h_i in dollars per hour by service type
    "Retail (node 1)": 25, "Retail (node 2)": 25, "Retail (node 3)": 25,
    "Premier": 27, "Business": 30, "Platinum": 33, "Priority Service": 33,
    "Consumer Loans": 23, "Online Banking": 23, "Telesales": 23,
    "EBO": 20, "Subanco": 20, "Case Quality": 20,
}
CALLS_PER_HOUR = 15      # abandonment penalty a_i = h_i / CALLS_PER_HOUR

# ------------------------------------------------------------------ test problem
N_SCALE = 100            # system parameter n
HORIZON_START = 9 * 60   # minutes after midnight, 9 A.M.
HORIZON_END = 17 * 60    # 5 P.M.
STEP = 5                 # minutes per grid interval
ARRIVAL_SCALE = 1.05     # multiplies the estimated arrival rates of the test problem
RHO_GRID_MIN, RHO_GRID_MAX, RHO_GRID_STEP = 0.50, 2.00, 0.0025   # search grid for rho^v
