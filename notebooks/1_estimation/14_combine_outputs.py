# # Prepare Data for Analysis
# ## Imports

# %load_ext autoreload
# %autoreload 2

from euraculus.data.map import DataMap
from euraculus.settings import DATA_DIR

data = DataMap(DATA_DIR)

# ## Sampling Summaries

# %%time
df_summary = data.load_selection_summary()
data.dump(df_summary, "analysis/df_summary.pkl")

# ## Index Estimates

# %%time
df_index = data.load_index_estimates()
data.dump(df_index, "analysis/df_index.pkl")

# ## Aggregate Statistics

# %%time
df_stats = data.load_estimation_summary()
data.dump(df_stats, "analysis/df_stats.pkl")

# ## Granular Estimates

# %%time
df_estimates = data.load_asset_estimates()
data.dump(df_estimates, "analysis/df_estimates.pkl")

# ## Consecutive History

# %%time
df_historic = data.load_nonoverlapping_historic()
data.dump(df_historic, "analysis/df_historic.pkl")

# ## Consecutive Forecast Data

# %%time
df_future = data.load_nonoverlapping_future()
data.dump(df_future, "analysis/df_future.pkl")
