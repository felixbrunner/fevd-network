# %% [markdown]
# # Raw Data Download
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
# - Index data from Yahoo! Finance
#
# Code to perform the steps is mainly in the `download.py` module

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

from euraculus.data import DataMap
from euraculus.download import WRDSDownloader, download_yahoo_data
from euraculus.settings import (
    DATA_DIR, 
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
)

# %% [markdown]
# ## Set up

# %% [markdown]
# ### WRDS Connection & DataMap

# %%
db = WRDSDownloader()
db._create_pgpass_file()

# %%
data = DataMap(DATA_DIR)

# %% [markdown]
# ### Timeframe

# %%
# first_year = 1925
first_year = FIRST_SAMPLING_DATE.year - 2
last_year = LAST_SAMPLING_DATE.year + 1

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
    data.dump(df, f"raw/crsp_{year}.pkl")
    if year % 5 == 0:
        print(f"    Year {year} done.")

# %% [markdown]
# ### Delisting Returns

# %%
# %%time
df_delist = db.download_delisting_returns()
data.dump(df_delist, "raw/delisting.pkl")

# %% [markdown]
# ### Descriptive Data

# %%
# %%time
df_descriptive = db.download_stocknames()
data.dump(df_descriptive, "raw/descriptive.pkl")

# %% [markdown]
# ### Industry Code Data

# %%
# %%time
df_sic = db.download_sic_table()
data.dump(df_sic, "raw/sic.pkl")
df_naics = db.download_naics_table()
data.dump(df_naics, "raw/naics.pkl")
df_gics = db.download_gics_table()
data.dump(df_gics, "raw/gics.pkl")

# %% [markdown]
# ## Download FF data

# %%
# %%time
df_ff = db.download_famafrench_factors()
data.dump(df_ff, "raw/ff_factors.pkl")

# %% [markdown]
# ## SPDR Trust SPY Index data

# %%
# %%time
df_spy = db.download_spy_data()
data.dump(df_spy, "raw/spy.pkl")

# %% [markdown]
# ## Yahoo data

# %% [markdown]
# ### CBOE Volatility Index (^VIX)

# %%
# %%time
ticker = "^VIX"
df_vix = download_yahoo_data(ticker)
data.dump(df_vix, f"raw/{ticker}.pkl")

# %% [markdown]
# ### US Dollar/USDX - Index - Cash (DX-Y.NYB)

# %%
# %%time
ticker = "DX-Y.NYB"
df_dxy = download_yahoo_data(ticker)
data.dump(df_dxy, f"raw/{ticker}.pkl")

# %% [markdown]
# ### Treasury Yield 10 Years (^TNX)

# %%
# %%time
ticker = "^TNX"
df_tnx = download_yahoo_data(ticker)
data.dump(df_tnx, f"raw/{ticker}.pkl")
