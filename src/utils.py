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


