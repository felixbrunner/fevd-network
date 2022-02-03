# %% [markdown]
# # 03 - Data Summary Stats
#
#
#
# ## Imports

# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import datetime as dt
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
%%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get samples
    df_back = data.load_sample(sampling_date.year, sampling_date.month, which="back", column="retadj")
    df_back -= df_rf.loc[df_back.index].values
    
    # calculate stats
    df_stats = pd.DataFrame(index=df_back.columns)
    df_stats["ret_excess"] = (1 + df_back).prod() - 1
    df_stats["var_annual"] = df_back.var() * 252
    
    # get excess return samples
    df_forward = data.load_sample(sampling_date.year, sampling_date.month, which="forward", column="retadj")
    df_forward -= df_rf.loc[df_forward.index].values
    
    # slice expanding window
    df_expanding_estimates = pd.DataFrame(index=df_forward.columns)
    for window_length in range(1, 13):
        end_date = sampling_date + relativedelta(months=window_length, day=31)
        df_window = df_forward[df_forward.index <= end_date]
    
        # calculate stats in window
        df_stats["ret_excess_next{}M".format(window_length)] = (1 + df_window).prod() - 1
        df_stats["var_annual_next{}M".format(window_length)] = df_window.var() * 252
    
    # store
    data.store(data=df_stats, path="samples/{0}{1:0=2d}/df_estimates.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed summary stats estimation at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)
    
    break

# %% [markdown]
# ## Indices summary stats

# %%
%%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get samples
    df_back = data.make_sample_indices(sampling_date.year, sampling_date.month, which="back")
    df_forward = data.make_sample_indices(sampling_date.year, sampling_date.month, which="forward")
    
    # calculate stats
    df_stats = pd.DataFrame(index=df_back.columns)
    df_stats["ret_excess"] = (1 + df_back).prod() - 1
    df_stats["var_annual"] = df_back.var() * 252
    
    # slice expanding window
    df_expanding_estimates = pd.DataFrame(index=df_forward.columns)
    for window_length in range(1, 13):
        end_date = sampling_date + relativedelta(months=window_length, day=31)
        df_window = df_forward[df_forward.index <= end_date]
    
        # calculate stats in window
        df_stats["ret_excess_next{}M".format(window_length)] = (1 + df_window).prod() - 1
        df_stats["var_annual_next{}M".format(window_length)] = df_window.var() * 252
    
    # store
    data.store(data=df_stats, path="samples/{0}{1:0=2d}/df_indices.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed summary stats estimation at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)
    
    break

# %%
