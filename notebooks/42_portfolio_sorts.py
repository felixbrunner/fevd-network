# # Crossectional Portfolio Sorts

# ## Imports

# %load_ext autoreload
# %autoreload 2
# # %matplotlib inline

# +
import pandas as pd
import numpy as np

import statsmodels.api as sm
# import statsmodels.formula.api as smf
import linearmodels as lm

import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from euraculus.data import DataMap
import kungfu as kf
# -

data = DataMap()
df_estimates = data.read("analysis/df_estimates.pkl")



def sort_portfolios(df_estimates, variable, num_portfolios):
    """"""
    labels = ["Low"] + list(np.arange(2, num_portfolios).astype(str)) + ["High"]
    out = df_estimates[variable].groupby("sampling_date").apply(lambda x: pd.qcut(x, q=num_portfolios, labels=labels)).rename(f"{variable}_portfolio")
    return df_estimates.reset_index().groupby(["sampling_date", out.values])


variable = "fevd_in_connectedness_weighted"
pfr = sort_portfolios(df_estimates, variable, 5)["ret_excess"].mean().unstack()#.cumsum().plot()
pfr1 = sort_portfolios(df_estimates, variable, 5)["ret_excess_next1M"].mean().unstack()#.cumsum().plot()
pfr12 = sort_portfolios(df_estimates, variable, 5)["ret_excess_next12M"].mean().unstack()#.cumsum().plot()

ax = pfr.mean().plot(label="formation")
ax.plot(pfr1.mean()*12, label="1M")
ax.plot(pfr12.mean(), label="12M")
ax.legend()
plt.show()

(pfr["High"] - pfr["Low"]).cumsum().plot()



n_port = 5
labels = ['Low'] + list(np.arange(2, n_port).astype(str)) + ['High']
df_estimates["last_mcap"].groupby('sampling_date').apply(lambda x: pd.qcut(x, q=n_port, labels=labels)).rename('sort')











def mean_portfolio_stat(df, sort_col, stat_col, n_port=5):
    labels = ['Low'] + list(np.arange(2,n_port).astype(str)) + ['High']
    pfs = df[sort_col].dropna().groupby('sampling_year').apply(lambda x: pd.qcut(x, q=n_port, labels=labels)).rename('sort')
    df_ = df[stat_col].to_frame().merge(pfs, how='left', left_index=True, right_on=['sampling_year', 'permno'])
    pf_rets = df_.groupby(['sampling_year', 'sort'])[stat_col].mean().unstack()
    return pf_rets


# +
outcomes = ['Total return (t)', 'Total return (t+1)', 'beta', 'beta_t+1', 'alpha', 'alpha_t+1', 'Volatility p.a. (t)', 'Volatility p.a. (t+1)']
rankings = ['beta', 'fev_others', 'in_connectedness', 'fev_all']#, 'VAR_intercept', 'in_lvl']

fig, axes = plt.subplots(len(outcomes), len(rankings), figsize=[20, 15])

for i, outcome in enumerate(outcomes):
    for j, ranking in enumerate(rankings):
        axes[i,j].plot(mean_portfolio_stat(df, ranking, outcome, 5).mean(), marker='o', markersize=10)
        axes[i,j].set_title('{} of portfolios sorted on {}'.format(outcome, ranking))
