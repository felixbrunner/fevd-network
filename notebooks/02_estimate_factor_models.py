# %% [markdown]
# # Factor models estimation & residuals
# This notebook contains:
# - monthly factor model estimation for:
#     - returns
#     - log variance
#
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import datetime as dt
import numpy as np

import pandas as pd
from dateutil.relativedelta import relativedelta

from euraculus.data import DataMap
from euraculus.factor import (
    CAPM,
    Carhart4FactorModel,
    FactorModel,
    FamaFrench3FactorModel,
    SPY1FactorModel,
    SPYVariance1FactorModel,
)
from euraculus.estimate import prepare_log_data
from euraculus.factor import estimate_models
from euraculus.utils import months_difference

# %% [markdown]
# ## Set up
# ### Data

# %%
data = DataMap("../data")
df_rf = data.load_rf()

# %% [markdown]
# ### Models

# %%
ret_models = {
    "spy_capm": SPY1FactorModel(data),
    "capm": CAPM(data),
    "ff3": FamaFrench3FactorModel(data),
    "c4": Carhart4FactorModel(data),
}

# %%
var_models = {
    "logvar_capm": SPYVariance1FactorModel(data),
}

# %% [markdown]
# ### Dates

# %%
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# %% [markdown]
# ## Standard Factor Models

# %% [markdown]
# ### Backward part

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get excess return samples
    df_historic = data.load_historic(sampling_date=sampling_date, column="retadj")
    df_historic -= df_rf.loc[df_historic.index].values

    # estimate models backwards
    df_estimates, df_residuals = estimate_models(ret_models, df_historic)

    # store
    data.store(
        data=df_residuals,
        path="samples/{:%Y-%m-%d}/historic_daily.csv".format(sampling_date),
    )
    data.store(
        data=df_estimates,
        path="samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date),
    )

    # increment monthly end of month
    print(
        "Completed historic return factor model estimation at {:%Y-%m-%d}".format(
            sampling_date
        )
    )
    sampling_date += relativedelta(months=1, day=31)

# %% [markdown]
# ### Forward part as expanding window

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date < last_sampling_date:
    # get excess return samples
    df_future = data.load_future(sampling_date=sampling_date, column="retadj")
    df_future -= df_rf.loc[df_future.index].values

    # slice expanding window
    df_expanding_estimates = pd.DataFrame(index=df_future.columns)
    for window_length in range(1, 13):
        if (
            months_difference(end_date=last_sampling_date, start_date=sampling_date)
            >= window_length
        ):
            end_date = sampling_date + relativedelta(months=window_length, day=31)
            df_window = df_future[df_future.index <= end_date]

            # estimate models in window
            df_estimates, df_residuals = estimate_models(ret_models, df_window)

            # collect
            df_estimates = df_estimates.add_suffix("_next{}M".format(window_length))
            df_expanding_estimates = df_expanding_estimates.join(df_estimates)

    # store
    data.store(
        data=df_expanding_estimates,
        path="samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date),
    )
    data.store(
        data=df_residuals,
        path="samples/{:%Y-%m-%d}/future_daily.csv".format(sampling_date),
    )

    # increment monthly end of month
    print(
        "Completed future return factor model estimation at {:%Y-%m-%d}".format(
            sampling_date
        )
    )
    sampling_date += relativedelta(months=1, day=31)

# %% [markdown]
# ## Variance Factor Models

# %% [markdown]
# ### Backward part

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get excess return samples
    df_var = data.load_historic(sampling_date=sampling_date, column="var")
    df_noisevar = data.load_historic(sampling_date=sampling_date, column="noisevar")
    df_historic = prepare_log_data(df_data=df_var, df_fill=df_noisevar)

    # estimate models backwards
    df_estimates, df_residuals = estimate_models(var_models, df_historic)

    # store
    data.store(
        data=df_residuals,
        path="samples/{:%Y-%m-%d}/historic_daily.csv".format(sampling_date),
    )
    data.store(
        data=df_estimates,
        path="samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date),
    )

    # increment monthly end of month
    print(
        "Completed historic variance factor model estimation at {:%Y-%m-%d}".format(
            sampling_date
        )
    )
    sampling_date += relativedelta(months=1, day=31)

# %% [markdown]
# ### Forward part as expanding window

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date < last_sampling_date:
    # get excess return samples
    df_var = data.load_future(sampling_date=sampling_date, column="var")
    df_noisevar = data.load_future(sampling_date=sampling_date, column="noisevar")
    df_future = prepare_log_data(df_data=df_var, df_fill=df_noisevar)

    # slice expanding window
    df_expanding_estimates = pd.DataFrame(index=df_future.columns)
    for window_length in range(1, 13):
        if (
            months_difference(end_date=last_sampling_date, start_date=sampling_date)
            >= window_length
        ):
            end_date = sampling_date + relativedelta(months=window_length, day=31)
            df_window = df_future[df_future.index <= end_date]

            # estimate models in window
            df_estimates, df_residuals = estimate_models(var_models, df_window)

            # collect
            df_estimates = df_estimates.add_suffix("_next{}M".format(window_length))
            df_expanding_estimates = df_expanding_estimates.join(df_estimates)

    # store
    data.store(
        data=df_expanding_estimates,
        path="samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date),
    )
    data.store(
        data=df_residuals,
        path="samples/{:%Y-%m-%d}/future_daily.csv".format(sampling_date),
    )

    # increment monthly end of month
    print(
        "Completed future variance factor model estimation at {:%Y-%m-%d}".format(
            sampling_date
        )
    )
    sampling_date += relativedelta(months=1, day=31)
