# ## Set up

# +
# %load_ext autoreload
# %autoreload 2

import pandas as pd

from euraculus.data.map import DataMap
from euraculus.utils.utils import months_difference
from dateutil.relativedelta import relativedelta

from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
    FORECAST_WINDOWS,
    INDICES,
)
# -

# ## Set up
# ### Data

data = DataMap(DATA_DIR)
df_rf = data.load_rf()

# ## Indices summary stats

# %%time
sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # get samples
    historic_indices = data.load_historic_aggregates(sampling_date=sampling_date)[INDICES]
    future_indices = data.load_future_aggregates(sampling_date=sampling_date)[INDICES]

    # calculate stats
    df_stats = pd.DataFrame(index=historic_indices.columns)
    df_stats["ret_excess"] = (1 + historic_indices).prod() - 1
    df_stats["var_annual"] = historic_indices.var() * 252

    if sampling_date < LAST_SAMPLING_DATE:
        # slice expanding window
        df_expanding_estimates = pd.DataFrame(index=future_indices.columns)
        for window_length in FORECAST_WINDOWS:
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
