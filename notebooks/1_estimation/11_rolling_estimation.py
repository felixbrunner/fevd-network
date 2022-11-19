# %% [markdown]
# # Rolling Factor FEVD estimation
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from euraculus.models.covariance import GLASSO, AdaptiveThresholdEstimator
from euraculus.data.map import DataMap
from euraculus.network.fevd import FEVD
from euraculus.models.var import FactorVAR

from euraculus.models.estimate import (
    load_estimation_data,
    estimate_fevd,
)
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
    FACTORS,
    VAR_GRID,
    COV_GRID,
    HORIZON,
)

# %% [markdown]
# ## Setup

# %%
data = DataMap(DATA_DIR)

# %% [markdown]
# ## Estimation
# ### Test single period

# %%
sampling_date = FIRST_SAMPLING_DATE  # dt.datetime(year=2022, month=3, day=31)

# %%
# %%time
# load data
df_info, df_log_vola, df_factors = load_estimation_data(
    data=data, sampling_date=sampling_date
)

# estimate
var_data = df_log_vola
factor_data = df_factors[FACTORS]
var_cv, var, cov_cv, cov, fevd = estimate_fevd(
    var_data=var_data,
    factor_data=factor_data,
    var_grid=VAR_GRID,
    cov_grid=COV_GRID,
)
residuals = var.residuals(var_data=var_data, factor_data=factor_data)

# %% [markdown]
# ### Rolling Window


# %%
# %%time
sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # load data
    df_info, df_log_vola, df_factors = load_estimation_data(
        data=data, sampling_date=sampling_date
    )

    # estimate
    var_data = df_log_vola
    factor_data = df_factors[FACTORS]
    var_cv, var, cov_cv, cov, fevd = estimate_fevd(
        var_data=var_data,
        factor_data=factor_data,
        var_grid=VAR_GRID,
        cov_grid=COV_GRID,
    )
    residuals = var.residuals(var_data=var_data, factor_data=factor_data)

    # store estimates
    data.dump(data=var_data, path=f"samples/{sampling_date:%Y-%m-%d}/var_data.pkl")
    data.dump(
        data=factor_data, path=f"samples/{sampling_date:%Y-%m-%d}/factor_data.pkl"
    )
    data.dump(data=var_cv, path=f"samples/{sampling_date:%Y-%m-%d}/var_cv.pkl")
    data.dump(data=var, path=f"samples/{sampling_date:%Y-%m-%d}/var.pkl")
    data.dump(data=cov_cv, path=f"samples/{sampling_date:%Y-%m-%d}/cov_cv.pkl")
    data.dump(data=cov, path=f"samples/{sampling_date:%Y-%m-%d}/cov.pkl")
    data.dump(data=fevd, path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")
    data.dump(data=residuals, path=f"samples/{sampling_date:%Y-%m-%d}/residuals.pkl")

    # increment monthly end of month
    print(f"Completed estimation at {sampling_date:%Y-%m-%d}")
    sampling_date += TIME_STEP
