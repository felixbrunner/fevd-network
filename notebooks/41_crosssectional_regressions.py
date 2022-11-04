# # Crossectional Regression Analysis

# ## Imports

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

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

# +
# import sys
# sys.path.append('../')
# sys.path.append('../../kungfu/')
# import kungfu as kf
# # import src

# +
# from euraculus.data import DataMap
# from euraculus.plot import (
#     plot_estimation_summary,
#     plot_regularisation_summary,
#     plot_network_summary,
#     plot_partial_r2,
# )
# -

# ## Settings

# +
# set decimals to be displayed in tables
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# default figure size
sns.set(rc={'figure.figsize': (17, 6)})
plt.rcParams['figure.figsize'] = [17, 8]

# plotting styles
# plt.style.use('fivethirtyeight')
plt.style.use('seaborn')
#plt.rcParams['figure.dpi'] = 80
# -

# ## Load data

data = DataMap()
df_estimates = data.read("analysis/df_estimates.pkl")


# +
# df_estimates.columns.tolist()
# -

def make_aggregate_regression_table(endogs, exog):
    """"""
    reg_table = kf.RegressionTable()
    
    for _, endog in endogs.iteritems():
        reg = lm.PanelOLS(dependent=endog,
            exog=exog,
            time_effects=True,
            entity_effects=False,
           )\
    .fit(cov_type='kernel')
        reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
    
    reg_table.columns = endogs.columns
        
    return reg_table


lm.PanelOLS(dependent=df_estimates['ret_excess_next12M'],
            exog=np.log(df_estimates[['fevd_in_connectedness', "capm_mktrf_next12M"]]),#, 'var_annual']]),
            time_effects=True,
            entity_effects=False,
           )\
    .fit(cov_type='kernel')

sns.jointplot(x=np.log(df_estimates['fevd_in_connectedness']), y=df_estimates['ret_excess_next12M'], kind='reg', height=8)
plt.show()


# ## Regressions

def get_var_order(exogs):
    vars_long = []
    vars_short = ['const']
    for reg in exogs:
        if type(reg) is str:
            vars_long += [reg]
        else:
            vars_long += reg
    for var in vars_long:
        if var not in vars_short:
            vars_short += [var]
    return vars_short


# +
exogs = ['in',
        # 'in_pct',
         #'out',
         ['in','capm_mktrf_est','ret_est'],
         ['in','capm_mktrf_est'],
         #['in','capm_mktrf_est'],
         #['in','capm_mktrf_est'],
         #['ff3f_mktrf_est','ff3f_smb_est','ff3f_hml_est'],
         #['c4f_mktrf_est','c4f_smb_est','c4f_hml_est','c4f_umd_est']
         ]
effects = [[],
           ['time'],
           ['entity'],
           ['time', 'entity']]

add_outputs = ['Time FE', 'Entity FE', 'N', 'R-squared', 'R-squared (inclusive)']

var_order = get_var_order(exogs)#['const','in','capm_mktrf_ana','ff3f_mktrf_ana','ff3f_smb_ana','ff3f_hml_ana','c4f_mktrf_ana','c4f_smb_ana','c4f_hml_ana','c4f_umd_ana']

# +
table_panel_regressions = kf.RegressionTable()
endog = 'ret12M_ana'

for exog in exogs:
    for fixed_effects in effects:
        table_panel_regressions = table_panel_regressions\
                                        .join_regression(df_merged.fit_panel_regression\
                                        (endog=endog, exog=exog, fixed_effects=fixed_effects, lag=0, cov_type='kernel'),\
                                        add_outputs=add_outputs)
table_panel_regressions\
    .change_descriptive_order(add_outputs)\
    .change_variable_order(var_order)
# -










