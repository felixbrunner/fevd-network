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

from euraculus.covar import GLASSO, AdaptiveThresholdEstimator
from euraculus.data import DataMap
from euraculus.fevd import FEVD
from euraculus.var import FactorVAR

from euraculus.estimate import (
    load_estimation_data,
    construct_pca_factors,
    estimate_fevd,
    describe_data,
    describe_var,
    describe_cov,
    describe_fevd,
    collect_var_estimates,
    collect_cov_estimates,
    collect_fevd_estimates,
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
sampling_date = dt.datetime(year=2021, month=12, day=31)

# %%
# %%time
# load data
df_info, df_log_mcap_vola, df_factors = load_estimation_data(
    data=data, sampling_date=sampling_date
)
df_pca = construct_pca_factors(df=df_log_mcap_vola, n_factors=1)
df_factors = df_factors.join(df_pca)

# estimate
var_data = df_log_mcap_vola
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
# FIRST_SAMPLING_DATE = dt.datetime(year=2009, month=6, day=30)

# %%
# %%time

sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # load data
    df_info, df_log_mcap_vola, df_factors = load_estimation_data(
        data=data, sampling_date=sampling_date
    )
    df_pca = construct_pca_factors(df=df_log_mcap_vola, n_factors=1)
    df_factors = df_factors.join(df_pca)

    # estimate
    var_data = df_log_mcap_vola
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

# %% [markdown]
# ## Extract results
# ### Test single period

# %%
sampling_date = dt.datetime(year=2021, month=12, day=31)

# %%
# %%time
# read data & estimates
var_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_data.pkl")
factor_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/factor_data.pkl")
var_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_cv.pkl")
var = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var.pkl")
cov_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov_cv.pkl")
cov = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov.pkl")
fevd = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")
residuals = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/residuals.pkl")

# collect estimation statistics
stats = describe_data(var_data)
stats.update(
    describe_var(var=var, var_cv=var_cv, var_data=var_data, factor_data=factor_data)
)
stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
stats.update(describe_fevd(fevd=fevd, horizon=HORIZON, data=var_data))
# stats = {key + '_factor': value for key, value in stats.items()}

# collect estimates
estimates = collect_var_estimates(var=var, var_data=var_data, factor_data=factor_data)
estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
estimates = estimates.join(
    collect_fevd_estimates(fevd=fevd, horizon=HORIZON, data=var_data)
)
# estimates = estimates.add_suffix("_factor")

# %% [markdown]
# ### Rolling Window


# %%
# FIRST_SAMPLING_DATE = dt.datetime(year=2012, month=4, day=30)

# %%
# %%time

sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:

    # load estimates
    var_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_data.pkl")
    factor_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/factor_data.pkl")
    var_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_cv.pkl")
    var = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var.pkl")
    cov_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov_cv.pkl")
    cov = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov.pkl")
    fevd = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")
    residuals = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/residuals.pkl")

    # collect estimation statistics
    stats = describe_data(var_data)
    stats.update(
        describe_var(var=var, var_cv=var_cv, var_data=var_data, factor_data=factor_data)
    )
    stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
    stats.update(describe_fevd(fevd=fevd, horizon=HORIZON, data=var_data))

    # collect estimates
    estimates = collect_var_estimates(
        var=var, var_data=var_data, factor_data=factor_data
    )
    estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
    estimates = estimates.join(
        collect_fevd_estimates(fevd=fevd, horizon=HORIZON, data=var_data)
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
