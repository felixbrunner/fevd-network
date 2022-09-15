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

from euraculus.data import DataMap
from euraculus.sampling import LargeCapSampler
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
)

# %% [markdown]
# ## Set up

# %% [markdown]
# ### Sampler

# %%
data = DataMap(DATA_DIR)
sampler = LargeCapSampler(datamap=data, n_assets=100, back_offset=12, forward_offset=12)

# %% [markdown]
# ## Conduct monthly sampling

# %% [markdown]
# ### Test single window

# %%
sampling_date = dt.datetime(year=2021, month=12, day=31)

# %%
# %%time
df_historic, df_future, df_summary = sampler.sample(sampling_date)
df_estimates = df_summary.loc[df_historic.index.get_level_values("permno").unique()]
df_estimates["ticker"] = df_historic["ticker"].unstack().iloc[-1, :].values
df_estimates["sic"] = (
    df_historic["comp_sic"]
    .fillna(df_historic["crsp_sic"])
    .unstack()
    .iloc[-1, :]
    .astype(int)
    .values
)
df_estimates["naics"] = (
    df_historic["comp_naics"]
    .fillna(df_historic["crsp_naics"])
    .unstack()
    .iloc[-1, :]
    .values
)
df_estimates["gics"] = df_historic["gic"].unstack().iloc[-1, :].values
df_estimates["sic_division"] = data.lookup_sic_divisions(df_estimates["sic"].values)
df_estimates["ff_sector"] = data.lookup_famafrench_sectors(df_estimates["sic"].values)
df_estimates["ff_sector_ticker"] = data.lookup_famafrench_sectors(
    df_estimates["sic"].values, return_tickers=True
)
df_estimates["gics_sector"] = data.lookup_gics_sectors(df_estimates["gics"].values)

# %% [markdown]
# ### Rolling window

# %%
# %%time
# perform monthly sampling and store samples locally
sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # get sample
    df_historic, df_future, df_summary = sampler.sample(sampling_date)
    df_estimates = df_summary.loc[df_historic.index.get_level_values("permno").unique()]
    df_estimates["ticker"] = df_historic["ticker"].unstack().iloc[-1, :].values
    df_estimates["sic"] = (
        df_historic["comp_sic"]
        .fillna(df_historic["crsp_sic"])
        .unstack()
        .iloc[-1, :]
        .astype(int)
        .values
    )
    df_estimates["naics"] = (
        df_historic["comp_naics"]
        .fillna(df_historic["crsp_naics"])
        .unstack()
        .iloc[-1, :]
        .values
    )
    df_estimates["gics"] = df_historic["gic"].unstack().iloc[-1, :].values
    df_estimates["sic_division"] = data.lookup_sic_divisions(df_estimates["sic"].values)
    df_estimates["ff_sector"] = data.lookup_famafrench_sectors(
        df_estimates["sic"].values
    )
    df_estimates["ff_sector_ticker"] = data.lookup_famafrench_sectors(
        df_estimates["sic"].values, return_tickers=True
    )
    df_estimates["gics_sector"] = data.lookup_gics_sectors(df_estimates["gics"].values)

    # dump
    data.store(
        df_historic,
        f"samples/{sampling_date:%Y-%m-%d}/historic_daily.csv",
    )
    data.store(
        df_future,
        f"samples/{sampling_date:%Y-%m-%d}/future_daily.csv",
    )
    data.store(
        df_summary,
        f"samples/{sampling_date:%Y-%m-%d}/selection_summary.csv",
    )
    data.store(
        df_estimates,
        f"samples/{sampling_date:%Y-%m-%d}/asset_estimates.csv",
    )

    # increment monthly end of month
    if sampling_date.month == 12:
        print(f"Done sampling year {sampling_date.year}.")
    sampling_date += TIME_STEP
