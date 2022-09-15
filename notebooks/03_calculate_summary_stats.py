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
from euraculus.utils import months_difference
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
)

# %% [markdown]
# ## Set up
# ### Data

# %%
data = DataMap(DATA_DIR)
df_rf = data.load_rf()

# %% [markdown]
# ## Assets summary stats

# %%
# %%time
sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # get samples
    df_historic = data.load_historic(sampling_date=sampling_date, column="retadj")
    df_historic -= df_rf.loc[df_historic.index].values

    # calculate stats
    df_stats = pd.DataFrame(index=df_historic.columns)
    df_stats["ret_excess"] = (1 + df_historic).prod() - 1
    df_stats["var_annual"] = df_historic.var() * 252

    if sampling_date < LAST_SAMPLING_DATE:
        # get excess return samples
        df_future = data.load_future(sampling_date=sampling_date, column="retadj")
        df_future -= df_rf.loc[df_future.index].values

        # slice expanding window
        df_expanding_estimates = pd.DataFrame(index=df_future.columns)
        for window_length in range(1, 13):
            if (
                months_difference(end_date=LAST_SAMPLING_DATE, start_date=sampling_date)
                >= window_length
            ):
                end_date = sampling_date + relativedelta(months=window_length, day=31)
                df_window = df_future[df_future.index <= end_date]

                # calculate stats in window
                df_stats[f"ret_excess_next{window_length}M"] = (
                    1 + df_window
                ).prod() - 1
                df_stats[f"var_annual_next{window_length}M"] = df_window.var() * 252

    # store
    data.store(
        data=df_stats,
        path=f"samples/{sampling_date:%Y-%m-%d}/asset_estimates.csv",
    )

    # increment monthly end of month
    print(f"Completed summary stats estimation at {sampling_date:%Y-%m-%d}")
    sampling_date += TIME_STEP

# %% [markdown]
# ## Indices summary stats

# %%
# %%time
sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # get samples
    historic_indices, future_indices = data.make_sample_indices(
        sampling_date=sampling_date
    )

    # calculate stats
    df_stats = pd.DataFrame(index=historic_indices.columns)
    df_stats["ret_excess"] = (1 + historic_indices).prod() - 1
    df_stats["var_annual"] = historic_indices.var() * 252

    if sampling_date < LAST_SAMPLING_DATE:
        # slice expanding window
        df_expanding_estimates = pd.DataFrame(index=future_indices.columns)
        for window_length in range(1, 13):
            if (
                months_difference(end_date=LAST_SAMPLING_DATE, start_date=sampling_date)
                >= window_length
            ):
                end_date = sampling_date + relativedelta(months=window_length, day=31)
                df_window = future_indices[future_indices.index <= end_date]

                # calculate stats in window
                df_stats[f"ret_excess_next{window_length}M"] = (
                    1 + df_window
                ).prod() - 1
                df_stats[f"var_annual_next{window_length}M".format(window_length)] = (
                    df_window.var() * 252
                )

    # store
    data.store(
        data=df_stats,
        path=f"samples/{sampling_date:%Y-%m-%d}/index_estimates.csv",
    )

    # increment monthly end of month
    print(f"Completed summary stats estimation at {sampling_date:%Y-%m-%d}")
    sampling_date += TIME_STEP
