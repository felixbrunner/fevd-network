# %% [markdown]
# # Data Summary Stats
#
#
#
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt

# %%
import pandas as pd
from dateutil.relativedelta import relativedelta

from euraculus.data import DataMap

# %% [markdown]
# ## Set up
# ### Data

# %%
data = DataMap("../data")
df_rf = data.load_rf()

# %% [markdown]
# ### Dates

# %%
# define timeframe
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# %% [markdown]
# ## Assets summary stats

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get samples
    df_historic = data.load_historic(sampling_date=sampling_date, column="retadj")
    df_historic -= df_rf.loc[df_historic.index].values

    # calculate stats
    df_stats = pd.DataFrame(index=df_historic.columns)
    df_stats["ret_excess"] = (1 + df_historic).prod() - 1
    df_stats["var_annual"] = df_historic.var() * 252

    if sampling_date < last_sampling_date:
        # get excess return samples
        df_future = data.load_future(sampling_date=sampling_date, column="retadj")
        df_future -= df_rf.loc[df_future.index].values

        # slice expanding window
        df_expanding_estimates = pd.DataFrame(index=df_future.columns)
        for window_length in range(1, 13):
            end_date = sampling_date + relativedelta(months=window_length, day=31)
            df_window = df_future[df_future.index <= end_date]

            # calculate stats in window
            df_stats["ret_excess_next{}M".format(window_length)] = (
                1 + df_window
            ).prod() - 1
            df_stats["var_annual_next{}M".format(window_length)] = df_window.var() * 252

    # store
    data.store(
        data=df_stats,
        path="samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date),
    )

    # increment monthly end of month
    print("Completed summary stats estimation at {:%Y-%m-%d}".format(sampling_date))
    sampling_date += relativedelta(months=1, day=31)

# %% [markdown]
# ## Indices summary stats

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get samples
    historic_indices, future_indices = data.make_sample_indices(
        sampling_date=sampling_date
    )

    # calculate stats
    df_stats = pd.DataFrame(index=historic_indices.columns)
    df_stats["ret_excess"] = (1 + historic_indices).prod() - 1
    df_stats["var_annual"] = historic_indices.var() * 252

    if sampling_date < last_sampling_date:
        # slice expanding window
        df_expanding_estimates = pd.DataFrame(index=future_indices.columns)
        for window_length in range(1, 13):
            end_date = sampling_date + relativedelta(months=window_length, day=31)
            df_window = future_indices[future_indices.index <= end_date]

            # calculate stats in window
            df_stats["ret_excess_next{}M".format(window_length)] = (
                1 + df_window
            ).prod() - 1
            df_stats["var_annual_next{}M".format(window_length)] = df_window.var() * 252

    # store
    data.store(
        data=df_stats,
        path="samples/{:%Y-%m-%d}/index_estimates.csv".format(sampling_date),
    )

    # increment monthly end of month
    print("Completed summary stats estimation at {:%Y-%m-%d}".format(sampling_date))
    sampling_date += relativedelta(months=1, day=31)

# %%
