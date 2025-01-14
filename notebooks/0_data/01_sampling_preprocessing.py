# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

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

from euraculus.data.map import DataMap
from euraculus.data.sampling import LargeCapSampler
from euraculus.settings import (
    DATA_DIR,
    STORAGE_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
    SAMPLING_VARIABLE,
    ESTIMATION_WINDOW,
    FORECAST_WINDOW,
    NUM_ASSETS,
    SPLIT_DATE,
)

# %% [markdown]
# ## Set up

# %% [markdown]
# ### Sampler

# %%
data = DataMap(DATA_DIR)
sampler = LargeCapSampler(
    datamap=data,
    n_assets=NUM_ASSETS,
    back_offset=ESTIMATION_WINDOW,
    forward_offset=FORECAST_WINDOW,
)

# %% [markdown]
# ## Conduct monthly sampling

# %% [markdown]
# ### Test single window

# %%
sampling_date = dt.datetime(year=1973, month=7, day=31)

# %%
# %%time
df_historic, df_future, df_summary = sampler.sample(
    sampling_date=sampling_date, sampling_variable=SAMPLING_VARIABLE
)
df_estimates = df_summary.loc[df_historic.index.get_level_values("permno").unique()]
df_estimates["ticker"] = df_historic["ticker"].unstack().iloc[-1, :].values
df_estimates["sic"] = (
    df_historic["comp_sic"]
    .fillna(df_historic["crsp_sic"])
    .unstack()
    .ffill()
    .iloc[-1, :]
    .astype(int)
    .values
)
df_estimates["naics"] = (
    df_historic["comp_naics"]
    .fillna(df_historic["crsp_naics"])
    .unstack()
    .ffill()
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
sampling_date = SPLIT_DATE #FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # get sample
    df_historic, df_future, df_summary = sampler.sample(
        sampling_date, sampling_variable=SAMPLING_VARIABLE
    )
    df_estimates = df_summary.loc[df_historic.index.get_level_values("permno").unique()]
    df_estimates["ticker"] = df_historic["ticker"].unstack().iloc[-1, :].values
    # df_estimates["cusip"] = df_historic["cusip"].unstack().iloc[-1, :].values
    df_estimates["sic"] = (
        df_historic["comp_sic"]
        .fillna(df_historic["crsp_sic"])
        .unstack()
        .ffill()
        .iloc[-1, :]
        .astype(int)
        .values
    )
    df_estimates["naics"] = (
        df_historic["comp_naics"]
        .fillna(df_historic["crsp_naics"])
        .unstack()
        .ffill()
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
        f"{STORAGE_DIR}/{sampling_date:%Y-%m-%d}/historic_daily.csv",
    )
    data.store(
        df_future,
        f"{STORAGE_DIR}/{sampling_date:%Y-%m-%d}/future_daily.csv",
    )
    data.store(
        df_summary,
        f"{STORAGE_DIR}/{sampling_date:%Y-%m-%d}/selection_summary.csv",
    )
    data.store(
        df_estimates,
        f"{STORAGE_DIR}/{sampling_date:%Y-%m-%d}/asset_estimates.csv",
    )

    # increment monthly end of month
    if sampling_date.month == 12:
        print(f"Done sampling year {sampling_date.year}.")
    sampling_date += TIME_STEP
