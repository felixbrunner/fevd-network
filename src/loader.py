import pandas as pd
import numpy as np
import scipy as sp

import datetime as dt

import src

def load_descriptive():
    '''Loads the descriptive data and returns pandas DataFrame.'''
    df_desc = pd.read_pickle('../data/raw/df_crsp_desc.pkl')
    df_desc = df_desc.astype({'permno': int, 'exchcd':int}) \
                     .set_index('permno')
    df_desc[['st_date','end_date']] = df_desc[['st_date','end_date']].apply(pd.to_datetime, format='%Y-%m-%d')
    return df_desc

def load_year(year, purpose='estimation'):
    '''Returns the data for a specific year.
    Can be either estimation or analysis data (next year).
    '''
    assert purpose in ['estimation', 'analysis'], \
        'purpose needs to be either estimation or analysis'
    df = pd.read_csv('../data/processed/yearly/df_{}_{}.csv'.format(purpose, str(year)))
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    return df

def load_factors(year=None):
    '''Returns factor data as pandas DataFrame.
    Can be limited to a single year.
    '''
    df = pd.read_pickle('../data/raw/df_ff_raw.pkl')
    df['date'] = pd.to_datetime(df['date'], yearfirst=True)
    df = df.set_index('date')
    if year is not None:
        df = df[df.index.year == year]
    return df

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



