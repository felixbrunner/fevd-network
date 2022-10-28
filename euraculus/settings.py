"""Settings module to define directories, sampling parameters, etc."""
from pathlib import Path
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np

# directories
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"

# rolling window
FIRST_SAMPLING_DATE = dt.datetime(year=1989, month=12, day=31)
LAST_SAMPLING_DATE = dt.datetime(year=2022, month=3, day=31)
TIME_STEP = relativedelta(months=1, day=31)

# sampling
SAMPLING_VARIABLE = "valuation_volatility"  # "mcap"
ESTIMATION_WINDOW = 12
FORECAST_WINDOW = 60

# estimation
FACTORS = ["crsp"]
VAR_GRID = {
    "alpha": np.geomspace(1e-4, 1e0, 13),
    "lambdau": np.geomspace(1e-2, 1e1, 13),
    #'gamma': np.geomspace(1e-2, 1e2, 15),
}
COV_GRID = {"alpha": np.geomspace(1e-3, 1e0, 25)}
HORIZON = 21
