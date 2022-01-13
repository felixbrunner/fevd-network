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
import wrds

import sys

sys.path.append("../")
import euraculus

# %% [markdown]
# ## Set up WRDS Connection

# %%
wrds_conn = wrds.Connection(wrds_username="felixbru")
# wrds_conn.create_pgpass_file()
# wrds_connection.close()

# %% [markdown]
# #### Explore database

# %%
libraries = wrds_conn.list_libraries()
library = "crsp"

# %%
library_tables = wrds_conn.list_tables(library=library)
table = "dsf"

# %%
table_description = wrds_conn.describe_table(library=library, table=table)

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
for year in range(1993, 2021):  # range(1960, 2020):
    df = euraculus.query.download_crsp_year(wrds_conn, year)
    df.to_pickle(path="../data/raw/crsp_{}.pkl".format(year))
    if year % 5 == 0:
        print("    Year {} done.".format(year))

# %% [markdown]
# ### Delisting Returns

# %%
df_delist = euraculus.query.download_delisting(wrds_conn)
df_delist.to_pickle(path="../data/raw/delisting.pkl")

# %% [markdown]
# ### Descriptive Data

# %%
df_descriptive = euraculus.query.download_descriptive(wrds_conn)
df_descriptive.to_pickle(path="../data/raw/descriptive.pkl")

# %% [markdown]
# ## Download FF data

# %% [markdown]
# ### SQL Query

# %%
df_ff = euraculus.query.download_famafrench(wrds_conn)
df_ff.to_pickle(path="../data/raw/ff_factors.pkl")

# %% [markdown]
# ## SPDR Trust SPY Index data

# %%
df_spy = euraculus.query.download_SPY(wrds_conn)
df_spy.to_pickle(path="../data/raw/spy.pkl")
