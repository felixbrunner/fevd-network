# # Prepare analysis
# ## Imports

# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
from euraculus.data import DataMap
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    # TIME_STEP,
)

# ## Load & prepare data

# %%time
data = DataMap()
df_estimates = data.read("analysis/df_estimates.pkl")
try:
    df_historic = data.read("analysis/df_historic.pkl")
    df_future = data.read("analysis/df_future.pkl")
except ValueError:
    df_historic = data.load_nonoverlapping_historic()
    df_future = data.load_nonoverlapping_future()
    data.dump(df_historic, "analysis/df_historic.pkl")
    data.dump(df_future, "analysis/df_future.pkl")

# ## Portfolio weights
# ### Long-only portfolio weights

df_weights = df_estimates.select_dtypes(float).groupby("sampling_date").apply(lambda x: x/x.sum())
df_weights["equal"] = df_weights.groupby("sampling_date").transform(lambda x: 1/x.count()).iloc[:, 0]

# ### Weight differnece residual weights

df_weights = df_weights.join(df_weights.sub(df_weights["mean_mcap"].values.reshape(-1,1)).add_suffix("_residual"))

# ### Filter out overleveraged weights

df_weights = df_weights.loc[:, (df_weights.min() >= -1) & (df_weights.max() <= 1)]

# ## Portfolio returns

# %%time
df_indices = df_future.join(df_weights.add_suffix("_weight"), on=["sampling_date", "permno"])
df_indices = (df_indices[df_weights.add_suffix("_weight").columns] * df_indices["retadj"].values.reshape(-1, 1)).groupby("date").sum()
df_indices.columns = df_indices.columns.str.replace(r"_weight$", "")

df_indices["mean_mcap"].cumsum().plot()

df_indices["fevd_out_connectedness_weighted"].cumsum().plot()

df_indices["fevd_out_connectedness_weighted_longshort"].cumsum().plot()

# this is not weighted yet
#
# fullout connectedness instead?

# ## Construct indices

df_index = pd.DataFrame()
df_index["equal"] = (df_historic_["retadj"] * df_historic_["equal_weight"]).groupby("date").sum()
df_index["mcap"] = (df_historic_["retadj"] * df_historic_["mcap_weight"]).groupby("date").sum()
df_index["network"] = (df_historic_["retadj"] * df_historic_["fevd_out_connectedness_weighted_weight"]).groupby("date").sum()
df_index["granular_equal"] = (df_historic_["retadj"] * df_historic_["granular_equal_weight"]).groupby("date").sum()
df_index["granular_network"] = (df_historic_["retadj"] * df_historic_["granular_network_weight"]).groupby("date").sum()
# df_index["granular_both"] = (df_historic_["retadj"] * df_historic_["granular_both_weight"]).groupby("date").sum()
df_index["resid"] = (df_historic_["capm_resid"] * df_historic_["fevd_out_connectedness_weighted_weight"]).groupby("date").sum()

df_index.corr()

df_index.rolling(250).sum().plot()

df_index.mean()

df_index.cumsum().plot()

df_index_f = pd.DataFrame()
df_index_f["equal"] = (df_future_["retadj"] * df_future_["equal_weight"]).groupby("date").sum()
df_index_f["mcap"] = (df_future_["retadj"] * df_future_["mcap_weight"]).groupby("date").sum()
df_index_f["network"] = (df_future_["retadj"] * df_future_["fevd_out_connectedness_weighted_weight"]).groupby("date").sum()
df_index_f["granular_equal"] = (df_future_["retadj"] * df_future_["granular_equal_weight"]).groupby("date").sum()
df_index_f["granular_network"] = (df_future_["retadj"] * df_future_["granular_network_weight"]).groupby("date").sum()
# df_index_f["granular_both"] = (df_future_["retadj"] * df_future_["granular_both_weight"]).groupby("date").sum()
df_index_f["resid"] = (df_future_["capm_resid"] * df_future_["fevd_out_connectedness_weighted_weight"]).groupby("date").sum()

df_index_f.corr()

df_index_f.rolling(250).sum().plot()

df_index_f.mean()

df_index_f.cumsum().plot()


