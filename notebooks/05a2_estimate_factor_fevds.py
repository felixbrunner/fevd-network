# %% [markdown]
# # Rolling Factor FEVD estimation
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt
from dateutil.relativedelta import relativedelta

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

# %% [markdown]
# ## Setup
# ### Parameters

# %%
# define timeframe
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# %%
factors = ["pca_1"]
var_grid = {
    "alpha": np.geomspace(1e-10, 1e0, 11),
    "lambdau": np.geomspace(1e-1, 1e1, 11),
    #'gamma': np.geomspace(1e-2, 1e2, 15),
}
cov_grid = {"alpha": np.geomspace(1e-3, 1e0, 25)}
horizon = 21

# %% [markdown]
# ### Data

# %%
data = DataMap("../data")

# %% [markdown]
# ## Test single period

# %%
sampling_date = dt.datetime(year=2021, month=12, day=31)

# %%
# %%time
# load data
df_info, df_log_mcap_vola, df_factors = load_estimation_data(data=data, sampling_date=sampling_date)
df_pca = construct_pca_factors(df=df_log_mcap_vola, n_factors=1)
df_factors = df_factors.join(df_pca)

# estimate
var_data = df_log_mcap_vola
factor_data = df_factors[factors]
var_cv, var, cov_cv, cov, fevd = estimate_fevd(var_data=var_data, factor_data=factor_data, var_grid=var_grid, cov_grid=cov_grid,)
residuals = var.residuals(var_data=var_data, factor_data=factor_data)

# collect estimation statistics
stats = describe_data(var_data)
stats.update(describe_var(var=var, var_cv=var_cv, var_data=var_data, factor_data=factor_data))
stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
stats.update(describe_fevd(fevd=fevd, horizon=horizon, data=var_data))
# stats = {key + '_factor': value for key, value in stats.items()}

# collect estimates
estimates = collect_var_estimates(var=var, var_data=var_data, factor_data=factor_data)
estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
estimates = estimates.join(collect_fevd_estimates(fevd=fevd, horizon=horizon, data=var_data))
# estimates = estimates.add_suffix("_factor")

# %% [markdown]
# ## Rolling Window


# %%
first_sampling_date = dt.datetime(year=2015, month=9, day=30)

# %%
# %%time

sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # load data
    df_info, df_log_mcap_vola, df_factors = load_estimation_data(data=data, sampling_date=sampling_date)
    df_pca = construct_pca_factors(df=df_log_mcap_vola, n_factors=1)
    df_factors = df_factors.join(df_pca)

    # estimate
    var_data = df_log_mcap_vola
    factor_data = df_factors[factors]
    var_cv, var, cov_cv, cov, fevd = estimate_fevd(var_data=var_data, factor_data=factor_data, var_grid=var_grid, cov_grid=cov_grid,)
    residuals = var.residuals(var_data=var_data, factor_data=factor_data)

    # collect estimation statistics
    stats = describe_data(var_data)
    stats.update(describe_var(var=var, var_cv=var_cv, var_data=var_data, factor_data=factor_data))
    stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
    stats.update(describe_fevd(fevd=fevd, horizon=horizon, data=var_data))
    # stats = {key + '_factor': value for key, value in stats.items()}

    # collect estimates
    estimates = collect_var_estimates(var=var, var_data=var_data, factor_data=factor_data)
    estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
    estimates = estimates.join(collect_fevd_estimates(fevd=fevd, horizon=horizon, data=var_data))
    # estimates = estimates.add_suffix("_factor")

    # store
    data.write(
        data=pd.Series(
            stats, index=pd.Index(stats, name="statistic"), name=sampling_date
        ),
        path="samples/{:%Y-%m-%d}/estimation_stats.csv".format(sampling_date),
    )
    data.store(
        data=estimates,
        path="samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date),
    )
    data.write(data=fevd, path="samples/{:%Y-%m-%d}/fevd.pkl".format(sampling_date))

    # increment monthly end of month
    print("Completed estimation at {:%Y-%m-%d}".format(sampling_date))
    sampling_date += relativedelta(months=1, day=31)
