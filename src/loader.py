import pandas as pd
import numpy as np
import scipy as sp
import datetime as dt

import src


def load_crsp_year(year):
    '''Reads a single year of raw crsp data.'''
    df = pd.read_pickle('../data/raw/crsp_{}.pkl'.format(year))
    return df

def load_year(sampling_year, data='returns', data_year=None):
    '''Returns the data for a specific year.
    Can be either estimation or analysis data (next year).
    '''
    if not data_year:
        data_year = sampling_year
    try:
        df = pd.read_csv('../data/processed/annual/{}/{}_{}.csv'.format(sampling_year, data, data_year))
    except:
        raise Exception('Data type <{}> could not be found.'.format(data))
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    return df

def load_spy_returns(year=None):
    '''Returns annual SPY log vola data as pandas DataFrame.
    Can be limited to a single year.
    '''
    df = pd.read_csv('../data/processed/factors/spy_ret.csv')
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    if year is not None:
        df = df[df.index.year == year]
    return df

def load_spy_vola(year=None):
    '''Returns annual SPY log vola data as pandas DataFrame.
    Can be limited to a single year.
    '''
    df = pd.read_csv('../data/processed/factors/spy_vola.csv')
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    if year is not None:
        df = df[df.index.year == year]
    return df

def load_year_tickers(year):
    '''Returns a dictionary
    '''
    tickers = pd.read_csv('../data/processed/annual/{}/tickers.csv'.format(year))
    tickers = dict(tickers.set_index('permno'))
    return tickers

def load_descriptive():
    '''Loads the descriptive data and returns pandas DataFrame.'''
    df_desc = pd.read_csv('../data/processed/descriptive.csv')
    df_desc = df_desc.astype({'permno': int, 'exchcd':int}) \
                     .set_index('permno')
    df_desc[['namedt','nameendt']] = df_desc[['namedt','nameendt']].apply(pd.to_datetime, format='%Y-%m-%d')
    return df_desc

def load_rf(year=None):
    '''Returns risk-free rate data as pandas DataFrame.
    Can be limited to a single year.
    '''
    df = pd.read_csv('../data/processed/factors/rf.csv')
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    if year is not None:
        df = df[df.index.year == year]
    return df

def load_factors(year=None):
    '''Returns factor data as pandas DataFrame.
    Can be limited to a single year.
    '''
    df = pd.read_pickle('../data/raw/ff_factors.pkl').reset_index()
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    if year is not None:
        df = df[df.index.year == year]
    return df






# def load_descriptive():
#     '''Loads the descriptive data and returns pandas DataFrame.'''
#     df_desc = pd.read_pickle('../data/raw/df_crsp_desc.pkl')
#     df_desc = df_desc.astype({'permno': int, 'exchcd':int}) \
#                      .set_index('permno')
#     df_desc[['st_date','end_date']] = df_desc[['st_date','end_date']].apply(pd.to_datetime, format='%Y-%m-%d')
#     return df_desc




def load_year_all(year):
    '''Returns the estimation and analysis data for a specific year.'''
    df_est = load_year(year, purpose='estimation')
    df_ana = load_year(year, purpose='analysis')
    df_factors_est = load_factors(year=year)
    df_factors_ana = load_factors(year=year+1)
    return (df_est, df_ana, df_factors_est, df_factors_ana)

def select_factor_data(df_factors_est, df_factors_ana, selected_factors=[]):
    '''Returns a subset of factors as a pandas DataFrame.'''
    df_est = df_factors_est[selected_factors]
    df_ana = df_factors_ana[selected_factors]
    return (df_est, df_ana)



def load_year_vola(year, purpose='est'):
    '''Returns the data for a specific year.
    Can be either estimation or analysis data (next year).
    '''
    assert purpose in ['est', 'ana'], \
        'purpose needs to be either estimation or analysis'
    df = pd.read_csv('../data/processed/yearly/vola/df_{}_{}.csv'.format(purpose, str(year)))
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    return df

def load_year_all_vola(year):
    '''Returns the estimation and analysis data for a specific year.'''
    df_est = load_year_vola(year, purpose='est')
    df_ana = load_year_vola(year, purpose='ana')
    df_factors_est = load_factors(year=year)
    df_factors_ana = load_factors(year=year+1)
    return (df_est, df_ana, df_factors_est, df_factors_ana)


def load_spy(year=None, columns=None):
    df = pd.read_pickle('../data/raw/df_spy.pkl')
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    if year:
        df = df[df.index.year == year]
    if columns:
        df = df[columns]
    return df


#####

# def load_year(year, purpose='estimation'):
#     '''Returns the data for a specific year.
#     Can be either estimation or analysis data (next year).
#     '''
#     assert purpose in ['estimation', 'analysis'], \
#         'purpose needs to be either estimation or analysis'
#     df = pd.read_csv('../data/processed/yearly/df_{}_{}.csv'.format(purpose, str(year)))
#     df['date'] = pd.to_datetime(df['date'], yearfirst=True)
#     df = df.set_index('date')
#     return df