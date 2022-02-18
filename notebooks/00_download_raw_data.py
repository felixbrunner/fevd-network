# %% [markdown]
# # 00 - Raw Data Download
# ## Description
#
# This notebook downloads the daily stock file data from CRSP to output tables containing the following variables:
# - date
# - permno as unique identifier
# - mcap as shares outstanding times price
# - return
# - intraday extreme value volatility estimate $\bar{\sigma}^{2}_{i,t} = {0.3607}(p_{i,t}^{high}-p_{i,t}^{low})^{2}$ based on Parkinson (1980), where $p_{i,t}$ is the logarithm of the dollar price
#
# Additionally, the following data is downloaded:
# - Fama-French Factor data
# - SPDR TRUST S&P500 ETF ("SPY")
#
# Code to perform the steps is mainly in the `query.py` module

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from euraculus.download import WRDSDownloader
from euraculus.data import DataMap

# %% [markdown]
# ## Set up

# %% [markdown]
# ### WRDS Connection & DataMap

# %%
db = WRDSDownloader()
db._create_pgpass_file()

# %%
data = DataMap("../data")

# %% [markdown]
# ### Timeframe

# %%
first_year = 1993
last_year = 2021

# %% [markdown]
# #### Explore database

# %%
libraries = db.list_libraries()

# %%
library_tables = db.list_tables(library="crsp")

# %%
table_description = db.describe_table(library="crsp", table="dsf")

# %% [markdown]
# ## Download CRSP data

# %% [markdown]
# ### Daily stock data

# %% [markdown]
# EXCHCD:
# - 1: NYSE
# - 2: NYSE MKT
# - 3: NASDAQ
#
# SHRCD:
# - 10: Ordinary common share, no special status found
# - 11: Ordinary common share, no special status necessary

# %%
# %%time
for year in range(first_year, last_year + 1):
    df = db.download_crsp_year(year=year)
    data.write(df, "raw/crsp_{}.pkl".format(year))
    if year % 5 == 0:
        print("    Year {} done.".format(year))

# %% [markdown]
# ### Delisting Returns

# %%
# %%time
df_delist = db.download_delisting_returns()
data.write(df_delist, "raw/delisting.pkl")

# %% [markdown]
# ### Descriptive Data

# %%
# %%time
df_descriptive = db.download_stocknames()
data.write(df_descriptive, "raw/descriptive.pkl")

# %% [markdown]
# ## Download FF data

# %%
# %%time
df_ff = db.download_famafrench_factors()
data.write(df_ff, "raw/ff_factors.pkl")

# %% [markdown]
# ## SPDR Trust SPY Index data

# %%
# %%time
df_spy = db.download_spy_data()
data.write(df_spy, "raw/spy.pkl")

# %%
