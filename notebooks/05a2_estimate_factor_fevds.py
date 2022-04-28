# %% [markdown]
# # Rolling Factor FEVD estimation
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt

import numpy as np

# %%
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import GridSearchCV

from euraculus.covar import GLASSO, AdaptiveThresholdEstimator
from euraculus.data import DataMap
from euraculus.fevd import FEVD
from euraculus.var import FactorVAR

from euraculus.estimate import (
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

# %% [markdown]
# ### Data

# %%
data = DataMap("../data")

# %% [markdown]
# ### Dates

# %%
# define timeframe
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# %% [markdown]
# ### Hyperparamters

# %%
var_grid = {
    "alpha": np.geomspace(1e-5, 1e0, 11),
    "lambdau": np.geomspace(1e-2, 1e2, 13),
    #'gamma': np.geomspace(1e-2, 1e2, 15),
}
var_grid = {
    "alpha": np.geomspace(1e-6, 1e0, 13),
    "lambdau": np.geomspace(1e-1, 1e1, 11),
    #'gamma': np.geomspace(1e-2, 1e2, 15),
}

# cov_grid = {'delta': np.geomspace(0.5, 1, 11),
#             'eta': np.linspace(0, 2, 13)}

cov_grid = {"alpha": np.geomspace(1e-2, 1e0, 25)}

horizon = 21

# %%
# var_grid = {'alpha': [1e-2],
#             'lambdau': [1e-1],
#             #'gamma': np.geomspace(1e-2, 1e2, 15),
#             }

# # cov_grid = {'delta': np.geomspace(0.5, 1, 11),
# #             'eta': np.linspace(0, 2, 13)}

# cov_grid = {'alpha': [1e-1]}

# horizon = 21

# %% [markdown]
# ## Test single period

# %%
sampling_date = dt.datetime(year=2021, month=12, day=31)

# %%
# %%time
# load & preprocess data
df_var = data.load_historic(sampling_date=sampling_date, column="var")
df_noisevar = data.load_historic(sampling_date=sampling_date, column="noisevar")
df_spy_var = data.load_spy_data(series="var").loc[df_var.index]
mean_size = data.load_asset_estimates(
    sampling_date=sampling_date, columns="mean_size"
).values.squeeze()
df_var = data.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
df_spy_var = data.log_replace(df_spy_var, method="min")

# estimate var
var = FactorVAR(has_intercepts=True, p_lags=1)
var_cv = var.fit_adaptive_elastic_net_cv(
    var_data=df_var,
    factor_data=df_spy_var,
    grid=var_grid,
    return_cv=True,
    penalize_factors=False,
)
residuals = var.residuals(var_data=df_var, factor_data=df_spy_var)

# estimate covariance
cov_cv = GridSearchCV(
    GLASSO(max_iter=200),
    param_grid=cov_grid,
    cv=12,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
).fit(residuals)
# cov_cv = GridSearchCV(
#         AdaptiveThresholdEstimator(),
#         param_grid=cov_grid,
#         cv=12,
#         n_jobs=-1,
#         verbose=1,
#         return_train_score=True,
#     ).fit(residuals)
cov = cov_cv.best_estimator_

# create fevd
fevd = FEVD(var.var_1_matrix_, cov.covariance_)

# collect estimation statistics
stats = describe_data(df_var)
stats.update(describe_var(var=var, var_cv=var_cv, var_data=df_var, factor_data=df_spy_var))
stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
stats.update(describe_fevd(fevd=fevd, horizon=horizon, data=df_var))
stats = {key + '_factor': value for key, value in stats.items()}

# collect estimates
estimates = collect_var_estimates(var=var, data=df_var)
estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
estimates = estimates.join(
    collect_fevd_estimates(
        fevd=fevd, horizon=horizon, data=df_var, sizes=mean_size
    )
)
estimates = estimates.add_suffix("_factor")

# %% [markdown]
# ## Rolling Window


# %%
# %%time

sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # load & preprocess data
    df_var = data.load_historic(sampling_date=sampling_date, column="var")
    df_noisevar = data.load_historic(sampling_date=sampling_date, column="noisevar")
    df_spy_var = data.load_spy_data(series="var").loc[df_var.index]
    mean_size = data.load_asset_estimates(
        sampling_date=sampling_date, columns="mean_size"
    ).values.squeeze()
    df_var = data.prepare_log_variances(df_var=df_var, df_noisevar=df_noisevar)
    df_spy_var = data.log_replace(df_spy_var, method="min")

    # estimate var
    var = FactorVAR(has_intercepts=True, p_lags=1)
    var_cv = var.fit_adaptive_elastic_net_cv(
        var_data=df_var,
        factor_data=df_spy_var,
        grid=var_grid,
        return_cv=True,
        penalize_factors=False,
    )
    residuals = var.residuals(var_data=df_var,factor_data=df_spy_var)

    # estimate covariance
    cov_cv = GridSearchCV(
        GLASSO(max_iter=200),
        param_grid=cov_grid,
        cv=12,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    ).fit(residuals)
    cov = cov_cv.best_estimator_

    # create fevd
    fevd = FEVD(var.var_1_matrix_, cov.covariance_)
    
    # collect estimation statistics
    stats = describe_data(df_var)
    stats.update(describe_var(var=var, var_cv=var_cv, var_data=df_var, factor_data=df_spy_var))
    stats.update(describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
    stats.update(describe_fevd(fevd=fevd, horizon=horizon, data=df_var))
    stats = {key + '_factor': value for key, value in stats.items()}

    # collect estimates
    estimates = collect_var_estimates(var=var, data=df_var)
    estimates = estimates.join(collect_cov_estimates(cov=cov, data=residuals))
    estimates = estimates.join(
        collect_fevd_estimates(
            fevd=fevd, horizon=horizon, data=df_var, sizes=mean_size
        )
    )
    estimates = estimates.add_suffix("_factor")

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
    data.write(data=fevd, path="samples/{:%Y-%m-%d}/factorfevd.pkl".format(sampling_date))

    # increment monthly end of month
    print("Completed estimation at {:%Y-%m-%d}".format(sampling_date))
    sampling_date += relativedelta(months=1, day=31)
