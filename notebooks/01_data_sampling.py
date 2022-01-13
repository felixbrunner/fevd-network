# %% [markdown]
# # 01 - Data Sampling & Preparation
# This notebook ...
#
# Code is mainly saved in the `sampling.py` module

# %% [markdown]
# ## TO DO
# - Same permco can have multiple permno

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
import numpy as np
import datetime as dt
import sys

sys.path.append("../")
import euraculus

# %% [markdown]
# ## Process factor data

# %%
euraculus.sampling.preprocess_ff_factors()

# %%
# spy = pd.read_pickle("../data/raw/spy.pkl")

# %%
euraculus.sampling.preprocess_spy()

# %% [markdown]
# ## Process descriptive data

# %%
df_descriptive = pd.read_pickle("../data/raw/descriptive.pkl")
df_descriptive.to_csv("../data/processed/descriptive.csv")

# %% [markdown]
# ## Process asset data (monthly sampling)

# %%
# ## RUN THIS TO CREATE FILSYSTEM STRUCTURE

# import os
# path = '../data/processed/monthly/'
# for year in range(2020, 2021):
#     os.mkdir(path+str(year))
#     for month in range(1,13):
#         os.mkdir(path+str(year)+'/'+str(month))

# %%
# %%time
first_year = 1994
last_year = 2020
for year in range(first_year, last_year + 1):
    for month in range(1, 13):
        euraculus.sampling.preprocess(
            year, month=month, n_assets=100, months_back=12, months_forward=12
        )
    print("processed sampling year {}".format(year))

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# # OLD

# %% [markdown]
# ## Format asset data

# %%
# %%time
first_year = 1993  # 1960
last_year = 2019
for year in range(first_year, last_year + 1):
    euraculus.sampling.preprocess_year(year, last_year=(year == last_year))
    print("processed sampling year {}".format(year))

# %% [markdown]
# ## Load raw data

# %%
df_crsp_raw = pd.read_pickle("../data/raw/df_crsp_raw.pkl")

# %%
df_crsp_delist = pd.read_pickle("../data/raw/df_crsp_delist.pkl")

# %%
df_crsp_desc = pd.read_pickle("../data/raw/df_crsp_desc.pkl")

# %%
df_ff_raw = pd.read_pickle("../data/raw/df_ff_raw.pkl")

# %% [markdown]
# ## Format FF data

# %%
# edit data formats
df_ff_raw["date"] = pd.to_datetime(df_ff_raw["date"], yearfirst=True)

# declare index
df_ff_raw.set_index("date", inplace=True)

# %%
df_ff_raw.to_pickle(path="../data/interim/df_ff.pkl")

# %% [markdown]
# ## Transform CRSP data to tidy data format & adjust returns for delisting

# %%
df_crsp_tidy = df_crsp_raw.reset_index()
del df_crsp_raw

# %%
# edit data formats
df_crsp_tidy["date"] = pd.to_datetime(df_crsp_tidy["date"], yearfirst=True)
df_crsp_tidy[["permno"]] = df_crsp_tidy[["permno"]].astype(int)

df_crsp_delist["permno"] = df_crsp_delist["permno"].astype(int)
df_crsp_delist["date"] = pd.to_datetime(df_crsp_delist["date"], yearfirst=True)

# %%
# merge
df_crsp_tidy = df_crsp_tidy.merge(df_crsp_delist, how="left", on=["permno", "date"])

# %%
# adjusted returns (for delisting)
df_crsp_tidy["retadj"] = (1 + df_crsp_tidy["ret"].fillna(0)) * (
    1 + df_crsp_tidy["dlret"].fillna(0)
) - 1
df_crsp_tidy["retadj"] = df_crsp_tidy["retadj"].where(
    df_crsp_tidy["ret"].notna() | df_crsp_tidy["dlret"].notna()
)

# %%
# declare index & sort
df_crsp_tidy.set_index(["date", "permno"], inplace=True)
df_crsp_tidy = df_crsp_tidy.drop(columns=["index", "dlret"])
df_crsp_tidy = df_crsp_tidy.sort_index()

# %%
df_crsp_tidy.to_pickle(path="../data/interim/df_crsp_tidy.pkl")

# %% [markdown]
# ## Format descriptive data

# %%
df_crsp_desc["permno"] = df_crsp_desc["permno"].astype(int)
df_aux = df_crsp_desc.groupby("permno").last()

# %%
df_aux.to_pickle(path="../data/interim/df_aux.pkl")

# %% [markdown]
# ## Filter biggest Assets per Year

# %%
# TEMPORARY CELL
df_crsp_tidy = pd.read_pickle("../data/interim/df_crsp_tidy.pkl")
df_aux = pd.read_pickle("../data/interim/df_aux.pkl")

# %%
# parameters
N_LARGEST = 100
ESTIMATION_YEARS = 1
ANALYSIS_YEARS = 1

# %%
# select years
sample_years = list(df_crsp_tidy.index.get_level_values("date").year.unique())
if (df_crsp_tidy.index.get_level_values("date").year == sample_years[0]).sum() < (
    df_crsp_tidy.index.get_level_values("date").year == sample_years[1]
).sum() * 0.5:
    sample_years = sample_years[1:]

# %%
# select assets function # PREVIOUSLY USED
def select_assets(df_estimation, n_assets):
    year_obs = len(df_estimation["ret"].unstack())

    df_select = pd.DataFrame()
    df_select["full_year"] = (
        df_estimation["retadj"].groupby("permno").count() > year_obs * 0.99
    )
    df_select["size"] = (
        df_estimation["mcap"]
        .unstack()
        .sort_index()
        .fillna(method="ffill", limit=1)
        .tail(1)
        .squeeze()
    )
    df_select["size_rank"] = (
        df_select["size"].where(df_select["full_year"]).rank(ascending=False)
    )

    selected_assets = list(df_select.index[df_select["size_rank"] <= n_assets])
    return selected_assets


# %%
# select assets function
def select_assets(df_estimation, df_analysis, n_assets):
    year_obs = len(df_estimation["ret"].unstack())

    df_select = pd.DataFrame()
    df_select["full_obs"] = (
        df_estimation["retadj"].groupby("permno").count() > year_obs * 0.99
    )
    df_select["subsequent_obs"] = df_analysis["ret"].groupby("permno").count() > 0
    df_select["size"] = (
        df_estimation["mcap"]
        .unstack()
        .sort_index()
        .fillna(method="ffill", limit=1)
        .tail(1)
        .squeeze()
    )
    df_select["size_rank"] = (
        df_select["size"]
        .where(df_select["full_obs"])
        .where(df_select["subsequent_obs"])
        .rank(ascending=False)
    )

    selected_assets = list(df_select.index[df_select["size_rank"] <= n_assets])
    return selected_assets


# %%
# PREVIOUSLY USED
df_estimation_tidy = pd.Series(dtype="float", index=pd.MultiIndex.from_arrays([[], []]))
df_analysis_tidy = pd.Series(dtype="float", index=pd.MultiIndex.from_arrays([[], []]))
df_indices = pd.DataFrame()

for year in sample_years[ESTIMATION_YEARS - 1 : -ANALYSIS_YEARS]:
    # slice time dime dimension
    df_estimation = df_crsp_tidy[
        (df_crsp_tidy.index.get_level_values("date").year > year - ESTIMATION_YEARS)
        & (df_crsp_tidy.index.get_level_values("date").year <= year)
    ]
    df_analysis = df_crsp_tidy[
        (df_crsp_tidy.index.get_level_values("date").year > year)
        & (df_crsp_tidy.index.get_level_values("date").year <= year + ANALYSIS_YEARS)
    ]

    # slice assets
    selected_assets = select_assets(df_estimation, df_analysis, N_LARGEST)
    df_estimation = df_estimation[
        [i in selected_assets for i in df_estimation.index.get_level_values("permno")]
    ]
    df_analysis = df_analysis[
        [i in selected_assets for i in df_analysis.index.get_level_values("permno")]
    ]

    # output adjusted returns data
    df_estimation = df_estimation["retadj"].unstack().fillna(0)
    df_analysis = df_analysis["retadj"].unstack()
    df_descriptive = df_aux.loc[selected_assets]

    # save
    df_estimation.to_csv("../data/processed/yearly/df_estimation_" + str(year) + ".csv")
    df_analysis.to_csv("../data/processed/yearly/df_analysis_" + str(year) + ".csv")
    df_descriptive.to_csv(
        "../data/processed/yearly/df_descriptive_" + str(year) + ".csv"
    )

    # collect full timeline
    df_estimation_tidy = df_estimation_tidy.append(df_estimation.stack())
    df_analysis_tidy = df_analysis_tidy.append(df_analysis.stack())
    df_indices[year] = selected_assets

    print(year, dt.datetime.today())

# %%
# VOLA
df_estimation_tidy = pd.Series(dtype="float", index=pd.MultiIndex.from_arrays([[], []]))
df_analysis_tidy = pd.Series(dtype="float", index=pd.MultiIndex.from_arrays([[], []]))
df_indices = pd.DataFrame()

for year in sample_years[ESTIMATION_YEARS - 1 : -ANALYSIS_YEARS]:
    # slice time dime dimension
    df_estimation = df_crsp_tidy[
        (df_crsp_tidy.index.get_level_values("date").year > year - ESTIMATION_YEARS)
        & (df_crsp_tidy.index.get_level_values("date").year <= year)
    ]
    df_analysis = df_crsp_tidy[
        (df_crsp_tidy.index.get_level_values("date").year > year)
        & (df_crsp_tidy.index.get_level_values("date").year <= year + ANALYSIS_YEARS)
    ]

    # slice assets
    selected_assets = select_assets(df_estimation, df_analysis, N_LARGEST)
    df_estimation = df_estimation[
        [i in selected_assets for i in df_estimation.index.get_level_values("permno")]
    ]
    df_analysis = df_analysis[
        [i in selected_assets for i in df_analysis.index.get_level_values("permno")]
    ]

    # output adjusted returns data
    df_estimation = df_estimation["vola"].unstack().fillna(0)
    df_analysis = df_analysis["vola"].unstack()
    df_descriptive = df_aux.loc[selected_assets]

    # save
    df_estimation.to_csv("../data/processed/yearly/vola/df_est_" + str(year) + ".csv")
    df_analysis.to_csv("../data/processed/yearly/vola/df_ana_" + str(year) + ".csv")
    df_descriptive.to_csv(
        "../data/processed/yearly/vola/df_descriptive_" + str(year) + ".csv"
    )

    # collect full timeline
    df_estimation_tidy = df_estimation_tidy.append(df_estimation.stack())
    df_analysis_tidy = df_analysis_tidy.append(df_analysis.stack())
    df_indices[year] = selected_assets

    print(year, dt.datetime.today())

# %%
df_estimation_tidy.to_pickle(path="../data/processed/df_estimation_data.pkl")
df_analysis_tidy.to_pickle(path="../data/processed/df_analysis_data.pkl")
df_indices.to_pickle(path="../data/processed/df_indices_data.pkl")

# %%


# %%


# %%
df_estimation_tidy.unstack().apply(lambda x: x.autocorr(1)).hist()

# %%
np.log((df_estimation_tidy.replace(0, 1e-8) ** 1 * np.sqrt(250))).hist(bins=100)

# %%


# %%


# %%


# %%
