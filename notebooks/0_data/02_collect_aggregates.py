# ## Imports

# +
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
)
from tqdm import tqdm
from euraculus.data.preprocess import construct_index, count_obs, construct_normalized_vola_index
# -

# ## Setup

data = DataMap(DATA_DIR)

df_spy = data.load_spy_data()
df_vix = data.load_yahoo("^VIX")
df_dxy = data.load_yahoo("DX-Y.NYB")
df_tnx = data.load_yahoo("^TNX")

# ## Extract

# %%time
# extract aggregate information for each sample
sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    # load & set up
    df_historic = data.load_historic(sampling_date=sampling_date)
    index = df_historic.iloc[:, 0].unstack().index
    df_crsp = data.load_crsp_data(
        start_date=df_historic.index.get_level_values("date")[0],
        end_date=df_historic.index.get_level_values("date")[-1],
    )
    df_aggregates = pd.DataFrame(index=index)
    
    # simple indices
    df_aggregates["crsp_ew"] = construct_index(df_crsp, column="retadj", weighting_column=None, logs=False)
    df_aggregates["crsp_vw"] = construct_index(df_crsp, column="retadj", weighting_column="mcap", logs=False)
    df_aggregates["crsp_var_ew"] = construct_index(df_crsp, column="var", weighting_column=None, logs=False)
    df_aggregates["crsp_var_vw"] = construct_index(df_crsp, column="var", weighting_column="mcap", logs=False)
    df_aggregates["sample_ew"] = construct_index(df_historic, column="retadj", weighting_column=None, logs=False)
    df_aggregates["sample_vw"] = construct_index(df_historic, column="retadj", weighting_column="mcap", logs=False)
    df_aggregates["sample_var_ew"] = construct_index(df_historic, column="var", weighting_column=None, logs=False)
    df_aggregates["sample_var_vw"] = construct_index(df_historic, column="var", weighting_column="mcap", logs=False)
    
    # observation counts
    df_aggregates["crsp_num_ret"] = count_obs(df_crsp, column="retadj")
    df_aggregates["crsp_num_var"] = count_obs(df_crsp, column="var")
    df_aggregates["crsp_num_noisevar"] = count_obs(df_crsp, column="noisevar")
    df_aggregates["sample_num_ret"] = count_obs(df_historic, column="retadj")
    df_aggregates["sample_num_var"] = count_obs(df_historic, column="var")
    df_aggregates["sample_num_noisevar"] = count_obs(df_historic, column="noisevar")
    
    # normalized indices
    df_aggregates["crsp_vola"] = construct_normalized_vola_index(df_crsp, logs=False).rename("crsp_vola")
    df_aggregates["crsp_log_vola"] = construct_normalized_vola_index(df_crsp, logs=True).rename("crsp_log_vola")
    df_aggregates["sample_vola"] = construct_normalized_vola_index(df_historic, logs=False).rename("sample_vola")
    df_aggregates["sample_log_vola"] = construct_normalized_vola_index(df_historic, logs=True).rename("sample_log_vola")
    
    # spy
    df_aggregates["spy_ret"] = df_spy.reindex(index)["ret"]
    df_aggregates["spy_vola"] = np.sqrt(df_spy.reindex(index)["var"])
    df_aggregates["vix_ret"] = df_vix.reindex(index)["ret"]
    df_aggregates["vix_vola"] = np.sqrt(df_vix.reindex(index)["var"])
    df_aggregates["dxy_ret"] = df_dxy.reindex(index)["ret"]
    df_aggregates["dxy_vola"] = np.sqrt(df_dxy.reindex(index)["var"])
    df_aggregates["tnx_ret"] = df_tnx.reindex(index)["ret"]
    df_aggregates["tnx_vola"] = np.sqrt(df_tnx.reindex(index)["var"])
    
    # store
    data.store(
        df_aggregates,
        f"samples/{sampling_date:%Y-%m-%d}/aggregates.csv",
    )

    # increment monthly end of month
    if sampling_date.month == 12:
        print(f"Done sampling year {sampling_date.year}.")
    sampling_date += TIME_STEP













from sklearn.decomposition import PCA

# +
# %%time

df_indices = pd.DataFrame()
for year in tqdm(range(FIRST_SAMPLING_DATE.year - 1, LAST_SAMPLING_DATE.year + 1)):
    df_year = pd.DataFrame()

    # load data
    df_crsp = data.load_crsp_data(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
    df_var = df_crsp["var"].unstack()
    df_noisevar = df_crsp["noisevar"].unstack()
    df_ret = df_crsp["retadj"].unstack()
    df_mcap = df_crsp["mcap"].unstack()

    # process data
    df_var[df_var == 0] = df_noisevar
    df_weights = df_mcap.div(df_mcap.sum(axis=1), axis=0)

    # equally weighted indices
    df_year["var_ew"] = df_var.mean(axis=1)
    df_year["var_vw"] = df_var.mul(df_weights).sum(axis=1)
    df_year["ret_ew"] = df_ret.mean(axis=1)
    df_year["ret_vw"] = df_ret.mul(df_weights).sum(axis=1)
    df_year["num_var"] = df_var.count(axis=1)
    df_year["num_noisevar"] = df_noisevar.count(axis=1)
    df_year["num_ret"] = df_ret.count(axis=1)

    # append
    df_indices = df_indices.append(df_year)

    # print(f"finished index construction in year {year}")
# -

ax = np.log(df_indices["ret_ew"] + 1).cumsum().plot(label="cumulative return")
ax.plot(df_indices["ret_ew"] * 250, alpha=0.5, label="daily return (annualized)")
ax.plot(df_indices["var_ew"] * 1000, label="daily variance index * 1000")
ax.plot(df_indices["ret_ew"].rolling(250).std() * 250, label="rolling annual return")
ax.legend()

ax = df_indices["num_ret"].plot(label="return")
ax.plot(df_indices["num_var"], label="high-low variance")
ax.plot(df_indices["num_noisevar"], label="bid-ask variance")
ax.legend()


((df_indices["var_ew"] * 250) ** 0.5).plot()


