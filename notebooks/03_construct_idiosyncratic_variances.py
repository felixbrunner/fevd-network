# ## Imports

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import euraculus
# import kungfu as kf
import datetime as dt
from dateutil.relativedelta import relativedelta
from euraculus.data import DataMap
from euraculus.factor import FactorModel, SPY1FactorModel, CAPM, FamaFrench3FactorModel, Carhart4FactorModel, SPYVariance1FactorModel

# ## Set up
# ### Data

data = DataMap("../data")
df_spy = data.load_spy_data(series="var")

# ### Dates

# define timeframe
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# ## Construct CAPM idiosyncratic variances

# ### Backward part

# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # load betas
    df_var = data.load_sample(sampling_date.year, sampling_date.month, which="back", column="var")
    df_betas = data.load_estimates(date=sampling_date, names=["spy_capm_spy"])
    spy_data = df_spy.loc[df_var.index]
    
    # decompose
    df_decomposition = euraculus.factor.decompose_variance(
            df_var, df_betas, spy_data
        )
    df_decomposition = df_decomposition.loc[:, ["sys", "idio"]].add_prefix("var_")
    
    # store
    data.store(data=df_decomposition, path="samples/{0}{1:0=2d}/df_back.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed decomposition at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)

# ### Forward part as expanding window

# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # load betas
    df_var = data.load_sample(sampling_date.year, sampling_date.month, which="forward", column="var")
    spy_data = df_spy.loc[df_var.index]
    
    # slice expanding window
    df_expanding_decompositions = pd.DataFrame(index=df_var.unstack().index)
    for window_length in range(1, 13):
        end_date = sampling_date + relativedelta(months=window_length, day=31)
        df_window = df_var[df_var.index <= end_date]
        spy_window = spy_data[spy_data.index <= end_date]
        betas_window = data.load_estimates(date=sampling_date, names=["spy_capm_spy_next{}M".format(window_length)])
    
        # decompose
        df_decomposition = euraculus.factor.decompose_variance(
            df_window, betas_window, spy_window
        )
        df_decomposition = df_decomposition.loc[:, ["sys", "idio"]].add_prefix("var_").add_suffix("_next{}M".format(window_length))
            
        # collect
        df_expanding_decompositions = df_expanding_decompositions.join(df_decomposition)
    
    # store
    data.store(data=df_expanding_decompositions, path="samples/{0}{1:0=2d}/df_forward.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed factor model estimation at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)


