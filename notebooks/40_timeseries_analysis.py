# # Aggregate Regression Analysis

# ## Imports

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import pandas as pd
import numpy as np

import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import linearmodels as lm

import datetime as dt
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import seaborn as sns

from euraculus.data import DataMap
from euraculus.utils import average_correlation
from euraculus.settings import FIRST_SAMPLING_DATE, LAST_SAMPLING_DATE, TIME_STEP
import kungfu as kf
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

# ## Data
# ### Load & merge

# %%time
data = DataMap("../data")
df_stats = data.load_estimation_summary()
df_index = data.load_index_estimates()

df_analysis = df_index.unstack()
df_analysis.columns = pd.Index([var + "_" + idx for (var, idx) in df_analysis.columns])
df_analysis = df_analysis.join(df_stats)

# ### Add lags

for lag in range(1, 13):
    for index in ['ew', 'vw', 'spy']:
        df_analysis[f"ret_excess_next{lag}M_{index}_lagged"] = df_analysis[f"ret_excess_next{lag}M_{index}"].shift(1)
        df_analysis[f"var_annual_next{lag}M_{index}_lagged"] = df_analysis[f"var_annual_next{lag}M_{index}"].shift(1)
for index in ['ew', 'vw', 'spy']:
    df_analysis[f"ret_excess_{index}_lagged"] = df_analysis[f"ret_excess_{index}"].shift(1)
    df_analysis[f"var_annual_{index}_lagged"] = df_analysis[f"var_annual_{index}"].shift(1)

# ### Add average correlation

# +
# %%time
average_return_correlations = pd.Series(dtype=float)

sampling_date = FIRST_SAMPLING_DATE
while sampling_date <= LAST_SAMPLING_DATE:
    df = data.load_historic(sampling_date, "retadj")
    average_return_correlations[sampling_date] = average_correlation(df)
    sampling_date += TIME_STEP
    
    if sampling_date.month == 12:
        print("Completed calculation at {:%Y-%m-%d}".format(sampling_date))
# -

average_return_correlations.plot()

# ## Analysis

# Goyal/Welch data: [link](https://sites.google.com/view/agoyal145)

df_analysis.corr()


# +
# df_analysis.columns.tolist()
# -

def make_aggregate_regression_table(endogs: pd.DataFrame, exog: pd.DataFrame) -> kf.table.RegressionTable:
    """"""
    reg_table = kf.RegressionTable()
    
    for _, endog in endogs.iteritems():
        reg = sm.OLS(endog=endog, exog=exog, missing='drop')\
                    .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
    
    reg_table.columns = endogs.columns
        
    return reg_table


# ### Variance regressions

# +
variable = "var_annual"
index = "ew"

endogs = (
    # np.log(
            # (
            df_analysis[[f"{variable}_{index}",
                         f"{variable}_next1M_{index}",
                         f"{variable}_next3M_{index}",
                         f"{variable}_next6M_{index}",
                         f"{variable}_next9M_{index}",
                         f"{variable}_next12M_{index}",]]#.diff()
)
# )
exog = sm.add_constant(
    pd.concat([
        # df_analysis[f"{variable}_{index}"],
        # df_analysis["fevd_asymmetry"].diff(),
        df_analysis["fevd_avg_connectedness"].diff(),
        # df_analysis["fevd_concentration_out_connectedness"].diff(),
        # df_analysis["fevd_concentration_out_eigenvector_centrality"].diff(),
        # df_analysis["fevd_concentration_out_page_rank"].diff(),
        df_analysis[f"var_annual_{index}_lagged"].diff(),        
    ], axis=1)
)
table = make_aggregate_regression_table(endogs, exog)
table = table.change_row_labels({'const':'intercept',
                                 f"{variable}_{index}": "lag",
                         'fev_asymmetry':'Network asymmetry $a$',
                         'fevd_avg_connectedness':'Average network connectedness $C^{avg}$',
                                 "fevd_concentration_out_connectedness": "Network concentration",
                                 })\
                     .drop_second_index()
table.columns = pd.MultiIndex.from_arrays([['same']+5*['horizon (months)'], ['period']+[1, 3, 6, 9, 12]])
table
# -
# ### Return regressions

df_reg.plot(figsize=(20,8))

# +
variable = "ret_excess"
index = "spy"

endogs = df_analysis[[f"{variable}_{index}",
                      f"{variable}_next1M_{index}",
                      f"{variable}_next3M_{index}",
                      f"{variable}_next6M_{index}",
                      f"{variable}_next9M_{index}",
                      f"{variable}_next12M_{index}",
                     ]]
df_reg = df_analysis[[
            # "fev_asymmetry",
            "fevd_avg_connectedness",
            # "fev_concentration_out_connectedness",
            # "fev_concentration_out_eigenvector_centrality",
            # "fev_concentration_out_page_rank",
            # 'fev_amplification'
            # "var_annual_spy",
        ]].copy()
df_reg.loc[(df_reg["fevd_avg_connectedness"]<0.15).values, "fevd_avg_connectedness"] = 0

exog = sm.add_constant(df_reg
    # np.log(
        # df_analysis[[
        #     # "fev_asymmetry",
        #     "fevd_avg_connectedness",
        #     # "fev_concentration_out_connectedness",
        #     # "fev_concentration_out_eigenvector_centrality",
        #     # "fev_concentration_out_page_rank",
        #     # 'fev_amplification'
        #     "var_annual_spy",
        # ]]
    # )
)
# exog = sm.add_constant(np.log(df_analysis["fev_asymmetry"] * df_analysis["fev_avg_connectedness"] * df_analysis["fev_amplification"]))
# print(exog.corr())
table = make_aggregate_regression_table(endogs, exog)
table = table.change_row_labels({'const':'intercept',
                         'fev_asymmetry':'Network asymmetry $a$',
                         'fevd_avg_connectedness':'Average network connectedness $C^{avg}$',
                                 })\
                     .drop_second_index()
table.columns = pd.MultiIndex.from_arrays([['same']+5*['horizon (months)'], ['period']+[1, 3, 6, 9, 12]])
table
# -

(df_analysis[[f"ret_excess_next12M_vw"]]/12).plot()

df_analysis[[f"{variable}_{index}",
                      f"{variable}_next1M_{index}",
                      f"{variable}_next3M_{index}",
                      f"{variable}_next6M_{index}",
                      f"{variable}_next9M_{index}",
                      f"{variable}_next12M_{index}",]].mean()












# +
reg_table = kf.RegressionTable()

endog_var = "ret_excess"
endog_index = "ew"
exog = sm.add_constant(df_analysis[["fevd_asymmetry", "fevd_avg_connectedness"]])


reg_table = kf.RegressionTable()

reg = sm.OLS(endog=df_analysis[endog_var+"_"+endog_index], exog=exog, missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])

for h in [1, 3, 6, 9, 12]:
    y = df_analysis[endog_var+f"_next{h}M_"+endog_index]
    # X = sm.add_constant(df_analysis[x_vars])#, 'ret_back_12M_{}'.format(index)]])
    reg = sm.OLS(endog=df_analysis[endog_var+f"_next{h}M_"+endog_index],
                 exog=exog,
                 missing='drop').fit(cov_type='HAC',
                                     cov_kwds={'maxlags': h})
    reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
# table = reg_table.copy()
reg_table



# +
reg_table = kf.RegressionTable()

endog_var = "var_annual"
endog_index = "ew"
exog = sm.add_constant(df_analysis[["fev_asymmetry", "fev_avg_connectedness"]])


reg_table = kf.RegressionTable()
reg = sm.OLS(endog=np.sqrt(df_analysis[endog_var+"_"+endog_index]), exog=exog, missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
for h in [1, 3, 6, 9, 12]:
    y = df_analysis[endog_var+f"_next{h}M_"+endog_index]
    # X = sm.add_constant(df_analysis[x_vars])#, 'ret_back_12M_{}'.format(index)]])
    reg = sm.OLS(endog=np.sqrt(df_analysis[endog_var+f"_next{h}M_"+endog_index]),
                 exog=exog,
                 missing='drop').fit(cov_type='HAC',
                                     cov_kwds={'maxlags': h})
    reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
# table = reg_table.copy()
reg_table



# +
reg_table = kf.RegressionTable()

# ew
y_index = 'ew'

x_vars = ["fevd_asymmetry", "fevd_avg_connectedness"]
reg_table = kf.RegressionTable()
reg = sm.OLS(endog=df_analysis[f"ret_excess_{y_index}"], exog=sm.add_constant(df_analysis[x_vars]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
for h in range(1, 13):#[1, 3, 6, 9, 12]:
    y = df_analysis[f"ret_excess_next{h}M_{y_index}"]
    X = sm.add_constant(df_analysis[x_vars])#, 'ret_back_12M_{}'.format(index)]])
    reg = sm.OLS(endog=y, exog=X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h})
    reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
# table = reg_table.copy()
reg_table



# +
reg_table = kf.RegressionTable()

# ew
y_index = 'ew'
x_vars = ["fev_asymmetry", "fev_avg_connectedness"]
reg_table = kf.RegressionTable()
reg = sm.OLS(endog=df_analysis[f"ret_excess_{y_index}"], exog=sm.add_constant(np.log(df_analysis["fev_asymmetry"]*df_analysis["fev_avg_connectedness"])), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
for h in range(1, 13):#[1, 3, 6, 9, 12]:
    y = df_analysis[f"ret_excess_next{h}M_{y_index}"]
    X = sm.add_constant(np.log(df_analysis["fev_asymmetry"]*df_analysis["fev_avg_connectedness"]))#, 'ret_back_12M_{}'.format(index)]])
    reg = sm.OLS(endog=y, exog=X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h})
    reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
# table = reg_table.copy()
reg_table



# +
# spy
index = 'spy'
reg_table = kf.RegressionTable()
reg = sm.OLS(endog=df_merged['ret_back_12M_spy'], exog=sm.add_constant(df_merged[['fev_asymmetry']]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
for h in [1, 3, 6, 9, 12]:
    y = df_merged['ret_forward_{}M_{}'.format(h, index)]
    X = sm.add_constant(df_merged[['fev_asymmetry']])#, 'ret_back_12M_{}'.format(index)]])
    reg = sm.OLS(endog=y, exog=X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h})
    reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
    
table = table.append(reg_table)\
             .change_row_labels({'const':'intercept',
                                 'fev_asymmetry':'Network asymmetry $a$',
                                 })\
             .drop_second_index()
table.columns = pd.MultiIndex.from_arrays([['same']+5*['horizon (months)'], ['period']+[1, 3, 6, 9, 12]])
# table.export_to_latex('../reports/tables/index_returns_prediction.tex')
table

# +
fig, ax = plt.subplots(1,1)
y_index = "spy"

ax.plot(df_analysis[f"ret_excess_{y_index}"], label=f"ret_excess_{y_index}")
ax.plot(df_analysis["fevd_asymmetry"], label="fevd_asymmetry")
ax.plot(df_analysis["fevd_avg_connectedness"], label="fevd_avg_connectedness")
# ax.plot(df_analysis["fev_asymmetry"]*df_analysis["fev_avg_connectedness"], label="interaction")

# for h in [3, 6]:
#     ax.plot(df_analysis[f"ret_excess_next{h}M_{y_index}"], label=f"ret_excess_next{h}M_{y_index}")

ax.legend()
# -















# #### Additional Measures

# ##### Correlations

# +
corr_ret = pd.Series(index=pd.MultiIndex.from_product([[],[]]), dtype='float64')
corr_idiovar = pd.Series(index=pd.MultiIndex.from_product([[],[]]), dtype='float64')

for year in range(1994, 2021):
    for month in range(1, 13):
        corr_ret[(year, month)] = src.utils.average_correlation(src.loader.load_monthly_crsp(year, month, which='back', column='retadj'))
        corr_idiovar[(year, month)] = src.utils.average_correlation(src.loader.load_monthly_estimation_data(year, month, column='idiosyncratic'))      
# -

df_new_measures =pd.DataFrame(data={'fev_asymmetry': df_merged['fev_asymmetry'],
                                    'fec_connectedness': df_merged['fev_avg_connectedness_normalised'],
#                                     'asy': asy.values, # is identical
                                    'asy_off_diagonal': asy_drop.values,
                                    'corr_ret': corr_ret.values,
                                    'corr_idiovar': corr_idiovar.values,
                                   })

df_new_measures.corr()

df_new_measures.plot()

# ### asd

# +
reg_table = kf.RegressionTable()

# ew
index = 'ew'
reg_table = kf.RegressionTable()
reg = sm.OLS(endog=df_merged['ret_back_12M_ew'], exog=sm.add_constant(df_merged[['fev_asymmetry']]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
for h in [1, 3, 6, 9, 12]:
    y = df_merged['ret_forward_{}M_{}'.format(h, index)]
    X = sm.add_constant(df_merged[['fev_asymmetry']])#, 'ret_back_12M_{}'.format(index)]])
    reg = sm.OLS(endog=y, exog=X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h})
    reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
table = reg_table.copy()


# spy
index = 'spy'
reg_table = kf.RegressionTable()
reg = sm.OLS(endog=df_merged['ret_back_12M_spy'], exog=sm.add_constant(df_merged[['fev_asymmetry']]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
for h in [1, 3, 6, 9, 12]:
    y = df_merged['ret_forward_{}M_{}'.format(h, index)]
    X = sm.add_constant(df_merged[['fev_asymmetry']])#, 'ret_back_12M_{}'.format(index)]])
    reg = sm.OLS(endog=y, exog=X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h})
    reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
    
table = table.append(reg_table)\
             .change_row_labels({'const':'intercept',
                                 'fev_asymmetry':'Network asymmetry $a$',
                                 })\
             .drop_second_index()
table.columns = pd.MultiIndex.from_arrays([['same']+5*['horizon (months)'], ['period']+[1, 3, 6, 9, 12]])
table.export_to_latex('../reports/tables/index_returns_prediction.tex')
table
# -





# +
reg_table = kf.RegressionTable()

# contemporaneous regressions
reg = sm.OLS(endog=df_merged['ret_back_12M_ew'], exog=sm.add_constant(df_merged[['fev_asymmetry', 'var_back_12M_ew']]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
reg = sm.OLS(endog=df_merged['ret_back_12M_spy'], exog=sm.add_constant(df_merged[['fev_asymmetry', 'var_back_12M_spy']]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])

# predictive regressions
for index in ['ew', 'spy']:
    for h in [1, 3, 6, 9, 12]:
        y = df_merged['ret_forward_{}M_{}'.format(h, index)]
        X = sm.add_constant(df_merged[['fev_asymmetry', 'var_back_12M_{}'.format(index)]])#, 'ret_back_12M_{}'.format(index)]])
        reg = sm.OLS(endog=y, exog=X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h})
        reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])

# formatting
reg_table.change_row_labels({'const':'intercept',
                             'fev_asymmetry':'Network asymmetry$_{t-1Y:t}$',
                             'var_back_12M_ew':'Equally weighted variance$_{t-1Y:t}$',
                             'var_back_12M_spy':'SPY variance$_{t-1Y:t}$',
                             'ret_back_12M_ew':'Equally weighted return$_{t-1Y:t}$',
                             'ret_back_12M_spy':'SPY return$_{t-1Y:t}$'})\
         .change_column_labels({'(1)':'$R^{ew}_{t-1Y:t}$',
                                '(2)':'$R^{spy}_{t-1Y:t}$',
                                '(3)':'$R^{ew}_{t:t+1M}$',
                                '(4)':'$R^{ew}_{t:t+3M}$',
                                '(5)':'$R^{ew}_{t:t+6M}$',
                                '(6)':'$R^{ew}_{t:t+9M}$',
                                '(7)':'$R^{ew}_{t:t+12M}$',
                                '(8)':'$R^{spy}_{t:t+1M}$',
                                '(9)':'$R^{spy}_{t:t+3M}$',
                                '(10)':'$R^{spy}_{t:t+6M}$',
                                '(11)':'$R^{spy}_{t:t+9M}$',
                                '(12)':'$R^{spy}_{t:t+12M}$'})\
         .drop_second_index()
reg_table.columns = pd.MultiIndex.from_arrays([['contemporaneous']*2+['predictive']*10, reg_table.columns], names=['', 'horizon'])
    
reg_table.export_to_latex('../reports/tables/index_returns_prediction.tex')
reg_table
# -



# +
reg_table = kf.RegressionTable()

# contemporaneous regressions
reg = sm.OLS(endog=df_merged['var_back_12M_ew'], exog=sm.add_constant(df_merged[['fev_avg_connectedness_normalised', 'ret_back_12M_ew']]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])
reg = sm.OLS(endog=df_merged['var_back_12M_spy'], exog=sm.add_constant(df_merged[['fev_avg_connectedness_normalised', 'ret_back_12M_spy']]), missing='drop')\
                .fit(cov_type='HAC', cov_kwds={'maxlags': 1})
reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])

# predictive regressions
for index in ['ew', 'spy']:
    for h in [1, 3, 6, 9, 12]:
        y = df_merged['var_forward_{}M_{}'.format(h, index)]
        X = sm.add_constant(df_merged[['fev_avg_connectedness_normalised', 'var_back_12M_{}'.format(index), 'ret_back_12M_{}'.format(index)]])
        reg = sm.OLS(endog=y, exog=X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h})
        reg_table = reg_table.join_regression(reg, add_outputs=['R-squared', 'N'])

# formatting
reg_table.change_row_labels({'const':'intercept',
                             'fev_asymmetry':'Network asymmetry$_{t-1Y:t}$',
                             'var_back_12M_ew':'Equally weighted variance$_{t-1Y:t}$',
                             'var_back_12M_spy':'SPY variance$_{t-1Y:t}$',
                             'ret_back_12M_ew':'Equally weighted return$_{t-1Y:t}$',
                             'ret_back_12M_spy':'SPY return$_{t-1Y:t}$'})\
         .change_column_labels({'(1)':'$R^{ew}_{t-1Y:t}$',
                                '(2)':'$R^{spy}_{t-1Y:t}$',
                                '(3)':'$R^{ew}_{t:t+1M}$',
                                '(4)':'$R^{ew}_{t:t+3M}$',
                                '(5)':'$R^{ew}_{t:t+6M}$',
                                '(6)':'$R^{ew}_{t:t+9M}$',
                                '(7)':'$R^{ew}_{t:t+12M}$',
                                '(8)':'$R^{spy}_{t:t+1M}$',
                                '(9)':'$R^{spy}_{t:t+3M}$',
                                '(10)':'$R^{spy}_{t:t+6M}$',
                                '(11)':'$R^{spy}_{t:t+9M}$',
                                '(12)':'$R^{spy}_{t:t+12M}$'})\
         .drop_second_index()
reg_table.columns = pd.MultiIndex.from_arrays([['contemporaneous']*2+['predictive']*10, reg_table.columns], names=['', 'horizon'])
    
reg_table.export_to_latex('../reports/tables/index_returns_prediction.tex')
reg_table
