# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Collect outputs for analysis
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
from euraculus.utils.utils import months_difference
from dateutil.relativedelta import relativedelta

from euraculus.models.estimate_vec import (
    describe_data,
    describe_vec,
    describe_structure,
    collect_data_estimates,
    collect_vec_estimates,
    collect_structural_estimates,
)
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
    HORIZON,
    FORECAST_WINDOWS,
    FIRST_ESTIMATION_DATE,
)

# %% [markdown]
# ## Set up
# ### Data

# %%
data = DataMap(DATA_DIR)
df_rf = data.load_rf()

# %% [markdown]
# ## Extract aggregate statistics and asset-level estimates for each period
# ### Test a single period

# %%
sampling_date = FIRST_ESTIMATION_DATE  # dt.datetime(year=2022, month=3, day=31)

# %%
# %%time
# read data & estimates
df_historic = data.load_historic(sampling_date=sampling_date, column="retadj")
df_future = (
    data.load_future(sampling_date=sampling_date, column="retadj")
    if sampling_date < LAST_SAMPLING_DATE
    else None
)
vec_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_data.pkl")
factor_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_factor_data.pkl")
vec_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_cv.pkl")
vec = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec.pkl")
residuals = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_residuals.pkl")
weights = data.load_asset_estimates(
    sampling_date=sampling_date, columns=["mean_mcap"]
).values.reshape(-1, 1)
weights /= weights.mean()

# collect estimation statistics
stats = describe_data(vec_data)
stats.update(
    describe_vec(vec=vec, vec_cv=vec_cv, vec_data=vec_data, factor_data=factor_data)
)
stats.update(
    describe_structure(vec=vec, weights=None)
)

# # collect estimates
estimates = collect_data_estimates(
    vec_data, df_historic, df_future, df_rf, FORECAST_WINDOWS
)
estimates = estimates.join(
    collect_vec_estimates(vec=vec, vec_data=vec_data, factor_data=factor_data)
)
estimates = estimates.join(
    collect_structural_estimates(vec=vec, data=vec_data, weights=None)
)

# %% [markdown]
# ### Rolling Window


# %%
# %%time
sampling_date = FIRST_ESTIMATION_DATE
while sampling_date <= LAST_SAMPLING_DATE:

    # read data & estimates
    df_historic = data.load_historic(sampling_date=sampling_date, column="retadj")
    df_future = (
        data.load_future(sampling_date=sampling_date, column="retadj")
        if sampling_date < LAST_SAMPLING_DATE
        else None
    )
    vec_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_data.pkl")
    factor_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_factor_data.pkl")
    vec_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_cv.pkl")
    vec = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec.pkl")
    residuals = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/vec_residuals.pkl")
    weights = data.load_asset_estimates(
        sampling_date=sampling_date, columns=["mean_mcap"]
    ).values.reshape(-1, 1)
    weights /= weights.mean()

    # collect estimation statistics
    stats = describe_data(vec_data)
    stats.update(
        describe_vec(vec=vec, vec_cv=vec_cv, vec_data=vec_data, factor_data=factor_data)
    )
    stats.update(
        describe_structure(vec=vec, weights=None)
    )

    # # collect estimates
    estimates = collect_data_estimates(
        vec_data, df_historic, df_future, df_rf, FORECAST_WINDOWS
    )
    estimates = estimates.join(
        collect_vec_estimates(vec=vec, vec_data=vec_data, factor_data=factor_data)
    )
    estimates = estimates.join(
        collect_structural_estimates(vec=vec, data=vec_data, weights=None)
    )

    # store
    data.store(
        data=pd.Series(
            stats, index=pd.Index(stats, name="statistic"), name=sampling_date
        ),
        path=f"samples/{sampling_date:%Y-%m-%d}/estimation_stats.csv",
    )
    data.store(
        data=estimates,
        path=f"samples/{sampling_date:%Y-%m-%d}/asset_estimates.csv",
    )

    # increment monthly end of month
    print(f"Completed calculations at {sampling_date:%Y-%m-%d}")
    sampling_date += TIME_STEP
