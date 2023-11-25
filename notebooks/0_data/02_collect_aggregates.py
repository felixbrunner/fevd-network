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
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from euraculus.data.map import DataMap
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
    ESTIMATION_WINDOW,
    SPLIT_DATE,
)
from tqdm import tqdm
from euraculus.data.preprocess import construct_index, count_obs, construct_normalized_vola_index, prepare_log_data
from kungfu.plotting import add_recession_bars

# %% [markdown]
# ## Setup

# %%
data = DataMap(DATA_DIR)

# %%
df_rf = data.load_rf()
df_spy = data.load_spy_data()
df_vix = data.load_yahoo("^VIX")
df_dxy = data.load_yahoo("DX-Y.NYB")
df_tnx = data.load_yahoo("^TNX")

df_crsp_index = data.read("raw/crsp_index.pkl")
df_crsp_index.index = pd.to_datetime(df_crsp_index.index)

# %% [markdown]
# ## Extract historic index stats (monthly)

# %%
# %%time
# extract aggregate information for each sample
sampling_date = SPLIT_DATE #FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # load & set up
    df_historic = data.load_historic(sampling_date=sampling_date)
    index = df_historic.iloc[:, 0].unstack().index
    rf = df_rf.reindex(index).values.squeeze()
    df_crsp = data.load_crsp_data(
        start_date=df_historic.index.get_level_values("date")[0],
        end_date=df_historic.index.get_level_values("date")[-1],
    )
    df_historic_aggregates = pd.DataFrame(index=index)
    
    # simple indices
    df_historic_aggregates["crsp_ew"] = construct_index(df_crsp, column="retadj", weighting_column=None, logs=False).sub(rf)
    df_historic_aggregates["crsp_vw"] = construct_index(df_crsp, column="retadj", weighting_column="mcap", logs=False).sub(rf)
    df_historic_aggregates["crsp_var_ew"] = construct_index(df_crsp, column="var", weighting_column=None, logs=False)
    df_historic_aggregates["crsp_var_vw"] = construct_index(df_crsp, column="var", weighting_column="mcap", logs=False)
    df_historic_aggregates["sample_ew"] = construct_index(df_historic, column="retadj", weighting_column=None, logs=False).sub(rf)
    df_historic_aggregates["sample_vw"] = construct_index(df_historic, column="retadj", weighting_column="mcap", logs=False).sub(rf)
    df_historic_aggregates["sample_var_ew"] = construct_index(df_historic, column="var", weighting_column=None, logs=False)
    df_historic_aggregates["sample_var_vw"] = construct_index(df_historic, column="var", weighting_column="mcap", logs=False)
    # overwrite crsp indices directly from soiurce
    df_historic_aggregates["crsp_ew"] = df_crsp_index[df_crsp_index.index.isin(index)]["ewretx"]
    df_historic_aggregates["crsp_vw"] = df_crsp_index[df_crsp_index.index.isin(index)]["vwretx"]
    
    # observation counts
    df_historic_aggregates["crsp_num_ret"] = count_obs(df_crsp, column="retadj")
    df_historic_aggregates["crsp_num_var"] = count_obs(df_crsp, column="var")
    df_historic_aggregates["crsp_num_noisevar"] = count_obs(df_crsp, column="noisevar")
    df_historic_aggregates["sample_num_ret"] = count_obs(df_historic, column="retadj")
    df_historic_aggregates["sample_num_var"] = count_obs(df_historic, column="var")
    df_historic_aggregates["sample_num_noisevar"] = count_obs(df_historic, column="noisevar")
    
    # normalized indices
    df_historic_aggregates["crsp_vola"] = construct_normalized_vola_index(df_crsp, logs=False)
    df_historic_aggregates["crsp_log_vola"] = construct_normalized_vola_index(df_crsp, logs=True)
    df_historic_aggregates["sample_vola"] = construct_normalized_vola_index(df_historic, logs=False)
    df_historic_aggregates["sample_log_vola"] = construct_normalized_vola_index(df_historic, logs=True)
    
    # spy
    df_historic_aggregates["spy_ret"] = df_spy.reindex(index)["ret"].sub(rf)
    df_historic_aggregates["spy_vola"] = np.sqrt(df_spy.reindex(index)["var"])
    df_historic_aggregates["vix_ret"] = df_vix.reindex(index)["ret"].sub(rf)
    df_historic_aggregates["vix_vola"] = np.sqrt(df_vix.reindex(index)["var"])
    df_historic_aggregates["dxy_ret"] = df_dxy.reindex(index)["ret"].sub(rf)
    df_historic_aggregates["dxy_vola"] = np.sqrt(df_dxy.reindex(index)["var"])
    df_historic_aggregates["tnx_ret"] = df_tnx.reindex(index)["ret"].sub(rf)
    df_historic_aggregates["tnx_vola"] = np.sqrt(df_tnx.reindex(index)["var"])
    
    
    # store
    data.store(
        df_historic_aggregates,
        f"samples/{sampling_date:%Y-%m-%d}/historic_aggregates.csv",
    )

    # increment monthly end of month
    if sampling_date.month == 12:
        print(f"Done collecting year {sampling_date.year}.")
    sampling_date += TIME_STEP

# %% [markdown]
# ## Extract future index stats (monthly)

# %%
# %%time
# extract aggregate information for each sample
sampling_date = SPLIT_DATE #FIRST_SAMPLING_DATE
while sampling_date < LAST_SAMPLING_DATE:
    # load & set up
    df_future = data.load_future(sampling_date=sampling_date)
    index = df_future.iloc[:, 0].unstack().index
    rf = df_rf.reindex(index).values.squeeze()
    df_crsp = data.load_crsp_data(
        start_date=df_future.index.get_level_values("date")[0],
        end_date=df_future.index.get_level_values("date")[-1],
    )
    df_future_aggregates = pd.DataFrame(index=index)
    
    # simple indices
    df_future_aggregates["crsp_ew"] = construct_index(df_crsp, column="retadj", weighting_column=None, logs=False).sub(rf)
    df_future_aggregates["crsp_vw"] = construct_index(df_crsp, column="retadj", weighting_column="mcap", logs=False).sub(rf)
    df_future_aggregates["crsp_var_ew"] = construct_index(df_crsp, column="var", weighting_column=None, logs=False)
    df_future_aggregates["crsp_var_vw"] = construct_index(df_crsp, column="var", weighting_column="mcap", logs=False)
    df_future_aggregates["sample_ew"] = construct_index(df_future, column="retadj", weighting_column=None, logs=False).sub(rf)
    df_future_aggregates["sample_vw"] = construct_index(df_future, column="retadj", weighting_column="mcap", logs=False).sub(rf)
    df_future_aggregates["sample_var_ew"] = construct_index(df_future, column="var", weighting_column=None, logs=False)
    df_future_aggregates["sample_var_vw"] = construct_index(df_future, column="var", weighting_column="mcap", logs=False)
    # overwrite crsp indices directly from soiurce
    df_future_aggregates["crsp_ew"] = df_crsp_index[df_crsp_index.index.isin(index)]["ewretx"]
    df_future_aggregates["crsp_vw"] = df_crsp_index[df_crsp_index.index.isin(index)]["vwretx"]
    
    # observation counts
    df_future_aggregates["crsp_num_ret"] = count_obs(df_crsp, column="retadj")
    df_future_aggregates["crsp_num_var"] = count_obs(df_crsp, column="var")
    df_future_aggregates["crsp_num_noisevar"] = count_obs(df_crsp, column="noisevar")
    df_future_aggregates["sample_num_ret"] = count_obs(df_future, column="retadj")
    df_future_aggregates["sample_num_var"] = count_obs(df_future, column="var")
    df_future_aggregates["sample_num_noisevar"] = count_obs(df_future, column="noisevar")
    
    # normalized indices
    df_future_aggregates["crsp_vola"] = construct_normalized_vola_index(df_crsp, logs=False).rename("crsp_vola")
    df_future_aggregates["crsp_log_vola"] = construct_normalized_vola_index(df_crsp, logs=True).rename("crsp_log_vola")
    df_future_aggregates["sample_vola"] = construct_normalized_vola_index(df_future, logs=False).rename("sample_vola")
    df_future_aggregates["sample_log_vola"] = construct_normalized_vola_index(df_future, logs=True).rename("sample_log_vola")
    
    # spy
    df_future_aggregates["spy_ret"] = df_spy.reindex(index)["ret"].sub(rf)
    df_future_aggregates["spy_vola"] = np.sqrt(df_spy.reindex(index)["var"])
    df_future_aggregates["vix_ret"] = df_vix.reindex(index)["ret"].sub(rf)
    df_future_aggregates["vix_vola"] = np.sqrt(df_vix.reindex(index)["var"])
    df_future_aggregates["dxy_ret"] = df_dxy.reindex(index)["ret"].sub(rf)
    df_future_aggregates["dxy_vola"] = np.sqrt(df_dxy.reindex(index)["var"])
    df_future_aggregates["tnx_ret"] = df_tnx.reindex(index)["ret"].sub(rf)
    df_future_aggregates["tnx_vola"] = np.sqrt(df_tnx.reindex(index)["var"])
    
    # store
    data.store(
        df_future_aggregates,
        f"samples/{sampling_date:%Y-%m-%d}/future_aggregates.csv",
    )

    # increment monthly end of month
    if sampling_date.month == 12:
        print(f"Done collecting year {sampling_date.year}.")
    sampling_date += TIME_STEP

# %% [markdown]
# ## Construct daily indices

# %%
# %%time

df_indices = pd.DataFrame(index=pd.Index([], name="date"))
for year in tqdm(range(FIRST_SAMPLING_DATE.year - 1, LAST_SAMPLING_DATE.year + 1)):
    df_year = pd.DataFrame()

    # load data
    df_crsp = data.load_crsp_data(dt.datetime(year - 1, 1, 1), dt.datetime(year, 12, 31))
    df_noisevar = df_crsp["noisevar"].astype(float).replace(0, np.nan).unstack()
    df_var = df_crsp["var"].unstack()
    df_var[df_var == 0] = df_noisevar
    df_vola = np.sqrt(df_var.replace(0, np.nan))
    df_log_vola = np.log(df_vola)
    df_ret = df_crsp["retadj"].unstack()
    df_mcap = df_crsp["mcap"].clip(lower=0).unstack()
    
    # process
    df_var[df_var == 0] = df_noisevar
    df_weights = df_mcap.div(df_mcap.sum(axis=1), axis=0)

    # equally weighted indices
    df_year["var_ew"] = df_var.mean(axis=1)
    df_year["ret_ew"] = df_ret.mean(axis=1)
    df_year["vola_ew"] = df_vola.mean(axis=1)
    df_year["logvola_ew"] = df_log_vola.mean(axis=1)
    df_year["vola_ew_99"] = df_vola.clip(upper=df_vola.quantile(0.99, axis=1), axis=0).mean(axis=1)
    df_year["logvola_ew_99"] = df_log_vola.clip(upper=df_log_vola.quantile(0.99, axis=1), axis=0).mean(axis=1)

    # value weighted indices
    df_year["var_vw"] = df_var.mul(df_weights).sum(axis=1)
    df_year["ret_vw"] = df_ret.mul(df_weights).sum(axis=1)
    df_year["vola_vw"] = df_vola.mul(df_weights).sum(axis=1)
    df_year["logvola_vw"] = df_log_vola.mul(df_weights).sum(axis=1)
    df_year["vola_vw_99"] = df_vola.clip(upper=df_vola.quantile(0.99, axis=1), axis=0).mul(df_weights).sum(axis=1)
    df_year["logvola_vw_99"] = df_log_vola.clip(upper=df_log_vola.quantile(0.99, axis=1), axis=0).mul(df_weights).sum(axis=1)

    # counts
    df_year["num_var"] = df_var.count(axis=1)
    df_year["num_noisevar"] = df_noisevar.count(axis=1)
    df_year["num_ret"] = df_ret.count(axis=1)

    # truncate 
    df_year = df_year[dt.datetime(year, 1, 1):dt.datetime(year, 12, 31)]

    # standardized indices
    df_year["logvola_ew_std"] = (df_log_vola
        .sub(df_log_vola.rolling(window='365D', min_periods=126).mean().shift(21))
        .div(df_log_vola.rolling(window='365D', min_periods=126).std().replace(0, np.nan).shift(21))
        [dt.datetime(year, 1, 1):]
        .mean(axis=1, skipna=True))
    df_year["logvola_vw_std"] = (df_log_vola
        .sub(df_log_vola.rolling(window='365D', min_periods=126).mean().shift(21))
        .div(df_log_vola.rolling(window='365D', min_periods=126).std().replace(0, np.nan).shift(21))
        [dt.datetime(year, 1, 1):]
        .mul(df_weights[dt.datetime(year, 1, 1):]).sum(axis=1))
    df_year["logvola_ew_std_lvl"] = (df_log_vola
        .sub(df_log_vola.rolling(window='365D', min_periods=126).mean().shift(21))
        .div(df_log_vola.rolling(window='365D', min_periods=126).std().replace(0, np.nan).shift(21))
        [dt.datetime(year, 1, 1):]
        .mean(axis=1, skipna=True)
         + df_log_vola.rolling(window='365D', min_periods=126).mean().shift(21)[dt.datetime(year, 1, 1):].mean(axis=1, skipna=True)
         )
    df_year["logvola_vw_std_lvl"] = (df_log_vola
        .sub(df_log_vola.rolling(window='365D', min_periods=126).mean().shift(21))
        .div(df_log_vola.rolling(window='365D', min_periods=126).std().replace(0, np.nan).shift(21))
        [dt.datetime(year, 1, 1):]
        .mul(df_weights[dt.datetime(year, 1, 1):]).sum(axis=1)
        + df_log_vola.rolling(window='365D', min_periods=126).mean().shift(21)[dt.datetime(year, 1, 1):].mul(df_weights[dt.datetime(year, 1, 1):]).sum(axis=1)
        )


    df_inv_vola_weights = 1/df_log_vola.rolling(window='365D', min_periods=126).std().replace(0, np.nan).shift(21)
    df_inv_vola_weights = df_inv_vola_weights.div(df_inv_vola_weights.sum(axis=1), axis=0)
    df_year["logvola_inv"] = df_log_vola.mul(df_inv_vola_weights[dt.datetime(year, 1, 1):]).sum(axis=1)

    # append
    df_indices = df_indices.append(df_year)
    
df_indices = df_indices\
    .merge(df_rf, how="outer", left_index=True, right_index=True)\
    .merge(df_spy[["ret", "var"]].add_prefix("spy_"), how="outer", left_index=True, right_index=True) \
    .merge(df_vix[["Close", "ret", "var"]].add_prefix("vix_"), how="outer", left_index=True, right_index=True) \
    .merge(df_dxy[["Close", "ret", "var"]].add_prefix("dxy_"), how="outer", left_index=True, right_index=True) \
    .merge(df_tnx[["Close", "ret", "var"]].add_prefix("tnx_"), how="outer", left_index=True, right_index=True)

# store
data.store(
    df_indices,
    f"analysis/df_daily_indices.pkl",
)

# %%
(np.exp(df_indices[["logvola_ew", ]])*np.sqrt(252))[dt.datetime(1980,1,1):].plot()

# %%
(np.exp(df_indices[[ "logvola_inv"]])*np.sqrt(252))[dt.datetime(1980,1,1):].plot()

# %%
(np.exp(df_indices[[ "logvola_ew_std"]]))[dt.datetime(1973,1,1):].plot()

# %%
((df_indices[[ "logvola_ew_std"]]))[dt.datetime(1973,1,1):].plot()

# %%
(np.exp(df_indices[[ "logvola_ew_std_lvl"]]))[dt.datetime(1980,1,1):].plot()

# %%
(np.log(df_indices[[ "vix_Close"]]/100))[dt.datetime(1980,1,1):].plot()

# %%
(df_indices["vix_Close"]).corr((df_indices["logvola_ew_std"]))

# %%
np.log(df_indices["vix_Close"]).corr((df_indices["logvola_vw_std_lvl"]))

# %%

# %%
year = 2020

# %%
df_crsp = data.load_crsp_data(dt.datetime(year-1, 1, 1), dt.datetime(year, 12, 31))
df_noisevar = df_crsp["noisevar"].astype(float).replace(0, np.nan).unstack()
df_var = df_crsp["var"].replace(0, np.nan).unstack()
df_vola = np.sqrt(df_var).fillna(np.sqrt(df_noisevar))
df_log_vola = prepare_log_data(np.sqrt(df_var), np.sqrt(df_noisevar))
df_ret = df_crsp["retadj"].unstack()
df_mcap = df_crsp["mcap"].clip(lower=0).unstack()
df_weights = df_mcap.div(df_mcap.sum(axis=1), axis=0)

# %%
(df_log_vola
    .sub(df_log_vola.rolling(window='365D', min_periods=126).mean().shift(1))
    .div(df_log_vola.rolling(window='365D', min_periods=120).std().replace(0, np.nan).shift(1))
    [dt.datetime(year, 1, 1):]
    .mean(axis=1, skipna=True)
).plot()

# %%
(df_log_vola
    .sub(df_log_vola.rolling(window='365D', min_periods=126).mean().shift(1))
    .div(df_log_vola.rolling(window='365D', min_periods=120).std().replace(0, np.nan).shift(1))
    [dt.datetime(year, 1, 1):]
    .mul(df_weights[dt.datetime(year, 1, 1):]).sum(axis=1)
).plot()

# %%

# %%

# %%
