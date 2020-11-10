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



def make_cv_splitter(n_splits, n_series, t_periods):
    '''Returns a PredefinedSplit object for cross validation.'''
    
    # shapes
    length = t_periods // n_splits
    resid = t_periods % n_splits

    # build single series split
    single_series_split = []
    for i in range(n_splits-1, -1 , -1):
        single_series_split += length*[i]
        if i < resid:
            single_series_split += [i]
            
    # make splitter object
    split = n_series*single_series_split
    splitter = PredefinedSplit(split)
    return splitter


def autocorrcoef(X, lag=1):
    '''Returns the autocorrelation matrix with input number of lags.'''
    N = X.shape[1]
    autocorr = np.corrcoef(X[lag:], X.shift(lag)[lag:], rowvar=False)[N:, :N]
    return autocorr


def cov_to_corr(cov):
    '''Returns a correlation matrix corresponding to the input covariance matrix'''
    stds = np.sqrt(np.diag(cov)).reshape(-1, 1)
    std_prod = stds @ stds.T
    corr = cov / std_prod
    return corr


def map_column_to_ticker(df_timeseries, df_descriptive):
    '''Returns a dictionary {column_number: ticker}.'''
    column_to_permno = {k: v for (k, v) in zip(range(df_timeseries.shape[1]), df_timeseries.columns)}
    permno_to_ticker = {str(int(k)): v for (k, v) in zip(df_descriptive.permno, df_descriptive.ticker)}
    column_to_ticker = {i: permno_to_ticker[k] for i, k in enumerate(column_to_permno.values())}
    return column_to_ticker


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