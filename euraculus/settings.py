"""Settings module to define directories, sampling parameters, etc."""
from pathlib import Path
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt

# directories
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
STORAGE_DIR = "samples"

# sampling parameters
NUM_ASSETS = 100
SAMPLING_VARIABLE = "mcap_volatility"  # "mcap"
FIRST_SAMPLING_DATE = dt.datetime(year=1927, month=6, day=30)
LAST_SAMPLING_DATE = dt.datetime(year=2022, month=3, day=31)
TIME_STEP = relativedelta(months=1, day=31)

# windows in months
ESTIMATION_WINDOW = 12
FORECAST_WINDOWS = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60]
FORECAST_WINDOW = max(FORECAST_WINDOWS)

# fixed dates
AMEX_INCLUSION_DATE = dt.datetime(year=1962, month=7, day=31)
NASDAQ_INCLUSION_DATE = dt.datetime(year=1972, month=12, day=31)
SPLIT_DATE = NASDAQ_INCLUSION_DATE + ESTIMATION_WINDOW * TIME_STEP

# estimation
FACTORS = ["crsp_log_vola"]

VAR_GRID = {
    "alpha": np.geomspace(1e-4, 1e0, 13),
    "lambdau": np.geomspace(1e-2, 1e1, 13),
    #'gamma': np.geomspace(1e-2, 1e2, 15),
}
COV_GRID = {"alpha": np.geomspace(1e-3, 1e0, 50)}
SIGMA_GRID = {"alpha": np.geomspace(1e-6, 1e-2, 50)}

# analysis
HORIZON = 21
INDICES = [
    "crsp_ew",
    "crsp_vw",
    "sample_ew",
    "sample_vw",
    "spy_ret",
    "vix_ret",
    "dxy_ret",
    "tnx_ret",
]
# FEVD_TABLES = [
#     ("fevd", None),
#     ("fev", None),
#     ("fevd", weights),
#     ("fev", weights),
# ] # (tabe_name, weights??)

# colors for plotting
DARK_BLUE = "#014c63"
DARK_GREEN = "#3b5828"
DARK_RED = "#880000"
ORANGE = "#ae3e00"
DARK_YELLOW = "#ba9600"
PURPLE = "#663a82"
COLORS = [DARK_BLUE, DARK_GREEN, DARK_RED, DARK_YELLOW, ORANGE, PURPLE]

# sector colors
colors = plt.get_cmap("Paired")(np.linspace(0, 1, 12))
sectors = [
    "NoDur",
    "Durbl",
    "Manuf",
    "Enrgy",
    "Chems",
    "BusEq",
    "Telcm",
    "Utils",
    "Shops",
    "Hlth",
    "Money",
    "Other",
]
SECTOR_COLORS = {sector: color for sector, color in zip(sectors, colors)}
