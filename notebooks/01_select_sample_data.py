# %% [markdown]
# # Data Sampling & Preparation
# This notebook serves to perform the sampling of the largest companies by market capitalization at the end of each month.

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from euraculus.data import DataMap
from euraculus.sampling import LargeCapSampler

# %% [markdown]
# ## Set up

# %% [markdown]
# ### Sampler

# %%
data = DataMap("../data")
sampler = LargeCapSampler(datamap=data, n_assets=100, back_offset=12, forward_offset=12)

# %% [markdown]
# ### Timeframe

# %%
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# %% [markdown]
# ## Conduct monthly sampling

# %% [markdown]
# ### Test single window

# %%
df_estimates
sampling_date = dt.datetime(year=2021, month=12, day=31)
df_historic, df_future, df_summary = sampler.sample(sampling_date)
df_estimates = df_summary.loc[df_historic.index.get_level_values("permno").unique()]
df_estimates["ticker"] = df_historic["ticker"].unstack().iloc[-1, :].values
df_estimates["sic"] = df_historic["comp_sic"].fillna(df_historic["crsp_sic"]).unstack().iloc[-1, :].astype(int).values
df_estimates["naics"] = df_historic["comp_naics"].fillna(df_historic["crsp_naics"]).unstack().iloc[-1, :].values
df_estimates["gics"] = df_historic["gic"].unstack().iloc[-1, :].values
df_estimates["sic_division"] = data.lookup_sic_divisions(df_estimates["sic"].values)
df_estimates["ff_sector"] = data.lookup_famafrench_sectors(df_estimates["sic"].values)
df_estimates["ff_sector_ticker"] = data.lookup_famafrench_sectors(df_estimates["sic"].values, return_tickers=True)
df_estimates["gics_sector"] = data.lookup_gics_sectors(df_estimates["gics"].values)

# %% [markdown]
# ### Rolling window

# %%
# %%time
# perform monthly sampling and store samples locally
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get sample
    df_historic, df_future, df_summary = sampler.sample(sampling_date)
    df_estimates = df_summary.loc[df_historic.index.get_level_values("permno").unique()]
    df_estimates["ticker"] = df_historic["ticker"].unstack().iloc[-1, :].values
    df_estimates["sic"] = df_historic["comp_sic"].fillna(df_historic["crsp_sic"]).unstack().iloc[-1, :].astype(int).values
    df_estimates["naics"] = df_historic["comp_naics"].fillna(df_historic["crsp_naics"]).unstack().iloc[-1, :].values
    df_estimates["gics"] = df_historic["gic"].unstack().iloc[-1, :].values
    df_estimates["sic_division"] = data.lookup_sic_divisions(df_estimates["sic"].values)
    df_estimates["ff_sector"] = data.lookup_famafrench_sectors(df_estimates["sic"].values)
    df_estimates["ff_sector_ticker"] = data.lookup_famafrench_sectors(df_estimates["sic"].values, return_tickers=True)
    df_estimates["gics_sector"] = data.lookup_gics_sectors(df_estimates["gics"].values)

    # dump
    data.store(
        df_historic,
        "samples/{:%Y-%m-%d}/historic_daily.csv".format(sampling_date),
    )
    data.store(
        df_future,
        "samples/{:%Y-%m-%d}/future_daily.csv".format(sampling_date),
    )
    data.store(
        df_summary,
        "samples/{:%Y-%m-%d}/selection_summary.csv".format(sampling_date),
    )
    data.store(
        df_estimates,
        "samples/{:%Y-%m-%d}/asset_estimates.csv".format(sampling_date),
    )

    # increment monthly end of month
    if sampling_date.month == 12:
        print("Done sampling year {}.".format(sampling_date.year))
    sampling_date += relativedelta(months=1, day=31)
