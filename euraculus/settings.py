"""Settings module to define directories, sampling parameters, etc."""
from pathlib import Path
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt

# directories
ROOT_DIR = Path(__file__).parents[1]
# DATA_DIR = ROOT_DIR / "data-mc"
# OUTPUT_DIR = ROOT_DIR / "outputs-mc"
# DATA_DIR = ROOT_DIR / "data-vv"
# OUTPUT_DIR = ROOT_DIR / "outputs-vv"
# DATA_DIR = ROOT_DIR / "data-vola"
# OUTPUT_DIR = ROOT_DIR / "outputs-vola"
DATA_DIR = ROOT_DIR / "data-mc-idx"
OUTPUT_DIR = ROOT_DIR / "outputs-mc-idx"

STORAGE_DIR = "samples"

# sampling parameters
NUM_ASSETS = 100
SAMPLING_VARIABLE = "valvola" #"mcap_volatility"#, "mcap"
FIRST_SAMPLING_DATE = dt.datetime(year=1927, month=6, day=30)
LAST_SAMPLING_DATE = dt.datetime(year=2022, month=12, day=31)
TIME_STEP = relativedelta(months=1, day=31)

# windows in months
ESTIMATION_WINDOW = 12
FORECAST_WINDOWS = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60]
FORECAST_WINDOW = max(FORECAST_WINDOWS)

# fixed dates
AMEX_INCLUSION_DATE = dt.datetime(year=1962, month=7, day=31)
NASDAQ_INCLUSION_DATE = dt.datetime(year=1972, month=12, day=31)
SPLIT_DATE = NASDAQ_INCLUSION_DATE
FIRST_ESTIMATION_DATE = SPLIT_DATE + ESTIMATION_WINDOW * TIME_STEP
# LAST_SAMPLING_DATE = SPLIT_DATE + 2 * ESTIMATION_WINDOW * TIME_STEP
FIRST_ANALYSIS_DATE = FIRST_ESTIMATION_DATE + relativedelta(days=1)
LAST_ANALYSIS_DATE = dt.datetime(year=2022, month=3, day=31)

# estimation
FACTORS = ["logvola_ew_std"] #["logvola_ew_99"] #["crsp_log_vola"]

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
# SECTOR_COLORS["Money"] = 

# # PRISM
# SECTOR_COLORS = {
#     "NoDur": "#5F4690",
#     "Durbl": "#1D6996",
#     "Manuf": "#38A6A5",
#     "Enrgy": "#0F8554",
#     "Chems": "#73AF48",
#     "BusEq": "#EDAD08",
#     "Telcm": "#E17C05",
#     "Utils": "#CC503E",
#     "Shops": "#94346E",
#     "Hlth": "#6F4070",
#     "Money": "#994E95",
#     "Other": "#666666",
#     }

# # SAFE
# SECTOR_COLORS = {
#     "NoDur": "#88CCEE",
#     "Durbl": "#CC6677",
#     "Manuf": "#DDCC77",
#     "Enrgy": "#117733",
#     "Chems": "#332288",
#     "BusEq": "#AA4499",
#     "Telcm": "#44AA99",
#     "Utils": "#999933",
#     "Shops": "#882255",
#     "Hlth": "#661100",
#     "Money": "#6699CC",
#     "Other": "#888888",
#     }

# # BOLD
# SECTOR_COLORS = {
#     "NoDur": "#7F3C8D",
#     "Durbl": "#11A579",
#     "Manuf": "#3969AC",
#     "Enrgy": "#F2B701",
#     "Chems": "#E73F74",
#     "BusEq": "#80BA5A",
#     "Telcm": "#E68310",
#     "Utils": "#008695",
#     "Shops": "#CF1C90",
#     "Hlth": "#f97b72",
#     "Money": "#4b4b8f",
#     "Other": "#A5AA99",
#     }

# # VIVID
# SECTOR_COLORS = {
#     "NoDur": "#E58606",
#     "Durbl": "#5D69B1",
#     "Manuf": "#52BCA3",
#     "Enrgy": "#99C945",
#     "Chems": "#CC61B0",
#     "BusEq": "#24796C",
#     "Telcm": "#DAA51B",
#     "Utils": "#2F8AC4",
#     "Shops": "#764E9F",
#     "Hlth": "#ED645A",
#     "Money": "#CC3A8E",
#     "Other": "#A5AA99",
# }

# CUSTOM
SECTOR_COLORS = {
    "NoDur": "#99CCFF",
    "Durbl": "#0080FF",
    "Manuf": "#CCFF99",
    "Enrgy": "#4C9900",
    "Chems": "#FF99CC",
    "BusEq": "#FF3333",
    "Telcm": "#FFCC99",
    "Utils": "#FF8000",
    "Shops": "#E5CCFF",
    "Hlth": "#9933FF",
    "Money": "#E5E500",
    "Other": "#CC6600",
}
