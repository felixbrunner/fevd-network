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

from euraculus.models.estimate import (
    describe_data,
    describe_var,
    describe_cov,
    describe_fevd,
    collect_data_estimates,
    collect_var_estimates,
    collect_cov_estimates,
    collect_fevd_estimates,
)
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
    HORIZON,
    FORECAST_WINDOWS,
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
sampling_date = dt.datetime(year=1927, month=6, day=30)

# %%
# %%time
# read data & estimates
df_historic = data.load_historic(sampling_date=sampling_date, column="retadj")
df_future = (
    data.load_future(sampling_date=sampling_date, column="retadj")
    if sampling_date < LAST_SAMPLING_DATE
    else None
)
var_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_data.pkl")
factor_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/factor_data.pkl")
var_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_cv.pkl")
var = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var.pkl")
cov_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov_cv.pkl")
cov = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov.pkl")
fevd = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")
residuals = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/residuals.pkl")
weights = data.load_asset_estimates(
    sampling_date=sampling_date, columns=["mean_mcap"]
).values.reshape(-1, 1)
weights /= weights.sum()

# collect estimation statistics
stats = describe_data(var_data)
stats.update(
    describe_var(var=var, var_cv=var_cv, var_data=var_data, factor_data=factor_data)
)
stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
stats.update(describe_fevd(fevd=fevd, horizon=HORIZON, data=var_data, weights=weights))

# collect estimates
estimates = collect_data_estimates(df_historic, df_future, df_rf, FORECAST_WINDOWS)
estimates = estimates.join(
    collect_var_estimates(var=var, var_data=var_data, factor_data=factor_data)
)
estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
estimates = estimates.join(
    collect_fevd_estimates(fevd=fevd, horizon=HORIZON, data=var_data, weights=weights)
)

# %% [markdown]
# ### Rolling Window


# %%
# %%time

sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:

    # load estimates
    df_historic = data.load_historic(sampling_date=sampling_date, column="retadj")
    df_future = (
        data.load_future(sampling_date=sampling_date, column="retadj")
        if sampling_date < LAST_SAMPLING_DATE
        else None
    )
    var_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_data.pkl")
    factor_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/factor_data.pkl")
    var_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_cv.pkl")
    var = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var.pkl")
    cov_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov_cv.pkl")
    cov = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov.pkl")
    fevd = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")
    residuals = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/residuals.pkl")
    weights = data.load_asset_estimates(
        sampling_date=sampling_date, columns=["mean_mcap"]
    ).values.reshape(-1, 1)
    weights /= weights.sum()

    # collect aggregate statistics
    stats = describe_data(var_data)
    stats.update(
        describe_var(var=var, var_cv=var_cv, var_data=var_data, factor_data=factor_data)
    )
    stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
    stats.update(
        describe_fevd(fevd=fevd, horizon=HORIZON, data=var_data, weights=weights)
    )

    # collect asset-level estimates
    estimates = collect_data_estimates(df_historic, df_future, df_rf, FORECAST_WINDOWS)
    estimates = estimates.join(
        collect_var_estimates(var=var, var_data=var_data, factor_data=factor_data)
    )
    estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
    estimates = estimates.join(
        collect_fevd_estimates(
            fevd=fevd, horizon=HORIZON, data=var_data, weights=weights
        )
    )

    # store
    data.dump(
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
