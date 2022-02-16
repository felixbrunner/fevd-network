# %% [markdown]
# # Rolling FEVD estimation
# ## Imports

# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import numpy as np

import datetime as dt
from dateutil.relativedelta import relativedelta

import euraculus
from sklearn.model_selection import GridSearchCV
from euraculus.data import DataMap

from euraculus.var import VAR
from euraculus.covar import GLASSO, AdaptiveThresholdEstimator
from euraculus.fevd import FEVD

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Data

# %%
data = DataMap("../data")

# %%
option = "logvar_capm_resid" # "spy_capm_decomp"

# %% [markdown]
# ### Dates

# %%
# define timeframe
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# %% [markdown]
# ### Hyperparamters

# %%
var_grid = {'alpha': np.geomspace(1e-5, 1e0, 11),
            'lambdau': np.geomspace(1e-2, 1e2, 13),
            #'gamma': np.geomspace(1e-2, 1e2, 15),
            }

# cov_grid = {'delta': np.geomspace(0.5, 1, 11),
#             'eta': np.linspace(0, 2, 13)}

cov_grid = {'alpha': np.geomspace(1e-2, 1e0, 25)}

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
%%time
# load and transform idiosyncratic volatility data, load size data
if option == "spy_capm_decomp":
    df_idio_var = data.load_historic(sampling_date=sampling_date, column="var_idio")
    df_log_idio_var = euraculus.utils.log_replace(df_idio_var, method='min')
elif option == "logvar_capm_resid":
    df_log_idio_var = data.load_historic(sampling_date=sampling_date, column="logvar_capm_resid")
mean_size = data.load_asset_estimates(sampling_date=sampling_date, columns="mean_size").values.squeeze()

# estimate var
var = VAR(add_intercepts=True, p_lags=1)
var_cv = var.fit_adaptive_elastic_net_cv(df_log_idio_var, grid=var_grid, return_cv=True)
residuals = var.residuals(df_log_idio_var)

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
stats = euraculus.estimate.describe_data(df_log_idio_var)
stats.update(euraculus.estimate.describe_var(var=var, var_cv=var_cv, data=df_log_idio_var))
stats.update(euraculus.estimate.describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
stats.update(euraculus.estimate.describe_fevd(fevd=fevd, horizon=horizon, data=df_log_idio_var))

# collect estimates
estimates = euraculus.estimate.collect_var_estimates(var=var, data=df_log_idio_var)
estimates = estimates.join(euraculus.estimate.collect_cov_estimates(cov=cov, data=residuals))
estimates = estimates.join(euraculus.estimate.collect_fevd_estimates(fevd=fevd, horizon=horizon, data=df_log_idio_var, sizes=mean_size))

# %% [markdown]
# ## Rolling Window

# %%
%%time

sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # load and transform idiosyncratic volatility data, load size data
    if option == "spy_capm_decomp":
        df_idio_var = data.load_historic(sampling_date=sampling_date, column="var_idio")
        df_log_idio_var = euraculus.utils.log_replace(df_idio_var, method='min')
    elif option == "logvar_capm_resid":
        df_log_idio_var = data.load_historic(sampling_date=sampling_date, column="logvar_capm_resid")
    mean_size = data.load_asset_estimates(sampling_date=sampling_date, columns="mean_size").values.squeeze()

    # estimate var
    var = VAR(add_intercepts=True, p_lags=1)
    var_cv = var.fit_adaptive_elastic_net_cv(df_log_idio_var, grid=var_grid, return_cv=True)
    residuals = var.residuals(df_log_idio_var)

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
    stats = euraculus.estimate.describe_data(df_log_idio_var)
    stats.update(euraculus.estimate.describe_var(var=var, var_cv=var_cv, data=df_log_idio_var))
    stats.update(euraculus.estimate.describe_cov(cov=cov, cov_cv=cov_cv, data=residuals))
    stats.update(euraculus.estimate.describe_fevd(fevd=fevd, horizon=horizon, data=df_log_idio_var))

    # collect estimates
    estimates = euraculus.estimate.collect_var_estimates(var=var, data=df_log_idio_var)
    estimates = estimates.join(euraculus.estimate.collect_cov_estimates(cov=cov, data=residuals))
    estimates = estimates.join(euraculus.estimate.collect_fevd_estimates(fevd=fevd, horizon=horizon, data=df_log_idio_var, sizes=mean_size))
    
    # store
    data.store(data=pd.Series(stats).to_frame(), path="samples/{:%Y-%m-%d}/estimation_stats.csv".format(sampling_date))
    data.store(data=estimates, path="samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date))
    data.store(data=fevd, path="samples/{:%Y-%m-%d}/fevd.pkl".format(sampling_date))
    
    # increment monthly end of month
    print("Completed estimation at {:%Y-%m-%d}".format(sampling_date))
    sampling_date += relativedelta(months=1, day=31)
