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
# # Prepare Data for Analysis
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from euraculus.data.map import DataMap
from euraculus.settings import DATA_DIR

# %%
data = DataMap(DATA_DIR)

# %% [markdown]
# ## Sampling Summaries

# %%
# %%time
df_summary = data.load_selection_summary().sort_index().drop_duplicates()
data.dump(df_summary, "analysis/df_summary.pkl")

# %% [markdown]
# ## Index Estimates

# %%
# %%time
df_index = data.load_index_estimates().sort_index().drop_duplicates()
data.dump(df_index, "analysis/df_index.pkl")

# %% [markdown]
# ## Aggregate Statistics

# %%
# %%time
df_stats = data.load_estimation_summary().sort_index().drop_duplicates()
data.dump(df_stats, "analysis/df_stats.pkl")

# %% [markdown]
# ## Granular Estimates

# %%
# %%time
df_estimates = data.load_asset_estimates().sort_index().drop_duplicates()
data.dump(df_estimates, "analysis/df_estimates.pkl")

# %% [markdown]
# ## Consecutive History

# %%
# %%time
df_historic = data.load_nonoverlapping_historic().sort_index().drop_duplicates()
data.dump(df_historic, "analysis/df_historic.pkl")

# %% [markdown]
# ## Consecutive Forecast Data

# %%
# %%time
df_future = data.load_nonoverlapping_future().sort_index().drop_duplicates()
data.dump(df_future, "analysis/df_future.pkl")
