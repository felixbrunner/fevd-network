# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
import src

def lookup_ticker(ticker, year):
    '''Returns descriptive data for a given ticker in a given year'''
    data = src.loader.load_descriptive().reset_index().set_index('ticker')
    data = data[(data.namedt.dt.year <= year) & (data.nameendt.dt.year >= year)]
    data = data.loc[ticker]
    return data

def calculate_nonzero_shrinkage(array, benchmark_array):
    '''Calculates the shrinkage factor of the nonzero elements in an array
    with regards to a benchmark array of the same dimensions.
    '''
    mean_scaling = 1-abs(array[array!=0]).mean()/abs(benchmark_array[array!=0]).mean()
    return mean_scaling

def calculate_full_shrinkage(array, benchmark_array):
    '''Calculates the scaling factor of the nonzero elements in an array
    with regards to a benchmark array of the same dimensions.
    '''
    mean_shrinkage = 1-abs(array).mean()/abs(benchmark_array).mean()
    return mean_shrinkage

def autocorrcoef(X, lag=1):
    '''Returns the autocorrelation matrix with input number of lags.'''
    N = X.shape[1]
    autocorr = np.corrcoef(X[lag:], X.shift(lag)[lag:], rowvar=False)[N:, :N]
    return autocorr

def cov_to_corr(cov):
    '''Returns a correlation matrix corresponding to the input covariance matrix.'''
    stds = np.sqrt(np.diag(cov)).reshape(-1, 1)
    std_prod = stds @ stds.T
    corr = cov / std_prod
    return corr

def prec_to_pcorr(prec):
    '''Returns a partial correlation matrix corresponding
    to the input precision matrix.
    '''
    stds = np.sqrt(np.diag(prec)).reshape(-1, 1)
    std_prod = stds @ stds.T
    pcorr = -prec / std_prod
    np.fill_diagonal(pcorr, 1)
    return pcorr

def log_replace(df, method='mean', logs=True, **kwargs):
    '''Takes logarithms of input DataFrame and fills missing
    values with an input fill method.
    '''
    df_full = df.copy()
    
    # logarithms
    if logs:
        df_full = np.log(df_full)
    
    # fill missing
    if method == 'mean':
        df_full = df_full.fillna(df_full.mean())
    elif method == 'interpolate':
        df_full = df_full.interpolate()
    elif method == 'interpolate':
        df_full = df_full.ffill(**kwargs)
    elif method == 'min':
        df_full = df_full.fillna(value=df_full.min())
    df_full.fillna(0)
    return df_full

def matrix_asymmetry(M):
    '''Returns a (self-built) measure of matrix asymmetry.'''
    M_s = (M+M.T)/2 # symmetric_part
    M_a = (M-M.T)/2 # asymmetric_part
    asymmetry = np.linalg.norm(M_a) / np.linalg.norm(M_s)
    return asymmetry



# def make_cv_splitter(n_splits, n_series, t_periods):
#     '''Returns a PredefinedSplit object for cross validation.'''
    
#     # shapes
#     length = t_periods // n_splits
#     resid = t_periods % n_splits

#     # build single series split
#     single_series_split = []
#     for i in range(n_splits-1, -1 , -1):
#         single_series_split += length*[i]
#         if i < resid:
#             single_series_split += [i]
            
#     # make splitter object
#     split = n_series*single_series_split
#     splitter = PredefinedSplit(split)
#     return splitter




def map_column_to_ticker(df_timeseries, df_descriptive):
    '''Returns a dictionary {column_number: ticker}.'''
    column_to_permno = {k: v for (k, v) in zip(range(df_timeseries.shape[1]), df_timeseries.columns)}
    permno_to_ticker = {str(int(k)): v for (k, v) in zip(df_descriptive.permno, df_descriptive.ticker)}
    column_to_ticker = {i: permno_to_ticker[k] for i, k in enumerate(column_to_permno.values())}
    return column_to_ticker

def summarise_returns(df):
    '''Returns the total return and annualised variance for
    an input dataframe of monthly sampled data.
    '''
    df = df.unstack('permno')
    df[[('weight', permno) for permno in df['mcap'].columns]] = \
            df['mcap']/df['mcap'].sum(axis=1).values.reshape(-1, 1)
    
    # build indices
    df_aggregate = pd.DataFrame(index=df.index)
    df_aggregate['ew_ret'] = df['retadj'].mean(axis=1)
    df_aggregate['vw_ret'] = (df['retadj']*df['weight']).sum(axis=1)
    df_aggregate['ew_var'] = df['var'].mean(axis=1)
    df_aggregate['vw_var'] = (df['var']*df['weight']).sum(axis=1)
    
    df_stats = pd.DataFrame()
    df_stats['ret'] = (1+df['retadj']).prod()-1
    df_stats.loc['ew', 'ret'] = (1+df_aggregate['ew_ret']).prod()-1
    df_stats.loc['vw', 'ret'] = (1+df_aggregate['vw_ret']).prod()-1
    df_stats['var'] = df['retadj'].var()*252
    df_stats.loc['ew', 'var'] = df_aggregate['ew_ret'].var()*252
    df_stats.loc['vw', 'var'] = df_aggregate['vw_ret'].var()*252
    
    return df_stats