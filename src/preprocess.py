import numpy as np
import pandas as pd
import os
import src



def _has_all_days(df):
    '''Returns a Series of booleans to indicate whether permnos
    have all observations in dataframe.
    '''
    t_periods = df.reset_index().date.nunique()
    has_all_days = df['retadj']\
                        .groupby('permno')\
                        .count() > t_periods*0.99
    has_all_days = has_all_days.rename('has_all_days')
    return has_all_days

def _has_obs(df, permnos):
    '''Returns a Series of booleans to indicate whether there are
     observations in dataframe given an iterable of permnos.
    '''
    df = df.copy().reset_index()
    permnos_in_df = df.permno.unique()
    has_obs = [permno in permnos_in_df for permno in permnos]
    has_obs = pd.Series(has_obs, index=permnos, name='has_obs')
    return has_obs

def _last_size(df):
    '''Returns a series of last observed market capitalisation
    for all permnos.
    '''
    last_size = df['mcap']\
                    .unstack()\
                    .sort_index()\
                    .fillna(method='ffill', limit=1)\
                    .tail(1)\
                    .squeeze()\
                    .rename('last_size')
    return last_size

def _mean_size(df):
    '''Returns a series of mean market capitalisation
    for all permnos.
    '''
    mean_size = df['mcap']\
                    .unstack()\
                    .mean()\
                    .squeeze()\
                    .rename('mean_size')
    return mean_size

def _tickers(df):
    '''Returns a series of tickers for all permos in
    the dataframe.
    '''
    tickers = df['ticker']\
                    .unstack()\
                    .tail(1)\
                    .squeeze()\
                    .rename('ticker')
    return tickers

def summarise_crsp_year(year, consider_next=True):
    '''Returns a summaring dataframe where the index consists 
    of the permnos present in the dataframe.
    Columns are:
    - ticker
    - has_all_days
    - has_obs
    - last_size
    - mean_size
    - size_rank
    '''
    # setup
    df_this = src.loader.load_crsp_year(year).sort_index()
    if consider_next:
        df_next = src.loader.load_crsp_year(year+1).sort_index()
    else:
        df_next = df_this
    permnos = df_this.reset_index().permno.unique()

    # assemble
    df_summary = _tickers(df_this).to_frame()\
                    .join(_has_all_days(df_this))\
                    .join(_has_obs(df_next, permnos).rename('has_next_obs'))\
                    .join(_last_size(df_this))\
                    .join(_mean_size(df_this))
    
    # cross-sectional size rank
    df_summary['size_rank'] = df_summary['last_size']\
                                .where(df_summary['has_all_days'])\
                                .where(df_summary['has_next_obs'])\
                                .rank(ascending=False)
    
    return df_summary

def n_largest_permnos(year, n=100, consider_next=True):
    '''Returns a list of permnos of the n companies
    with the highest market capitalisation end of year.'''
    permnos = src.preprocess.summarise_crsp_year(year, consider_next)\
                    .sort_values('size_rank')\
                    .head(n)\
                    .reset_index()\
                    .permno\
                    .values\
                    .tolist()
    return permnos

def filter_permnos(df, permnos):
    '''Returns a dataframe that only contains the input permnos.'''
    df = df[df.index.isin(permnos, level='permno')]
    return df

def make_return_matrix(df):
    '''Returns a dataframe with daily returns.
    Index are dates, columns are permnos.
    '''
    df_returns = df['retadj'].unstack()
    return df_returns

def make_vola_matrix(df):
    '''Returns a dataframe with intraday volatility.
    Index are dates, columns are permnos.
    '''
    df_vola = df['vola'].unstack().fillna(0)
    df_vola = (df_vola.replace(0, 1e-8))
    return df_vola

def make_ticker_series(df):
    '''Returns a ticker series indexed by permno.'''
    permno_to_ticker = df['ticker']\
                            .unstack()\
                            .tail(1)\
                            .squeeze()\
                            .rename('permno_to_ticker')
    return permno_to_ticker

def preprocess_year(year, n_assets=100, subsequent_years=1, last_year=False):
    '''Preprocesses a single sampling date in the following way:
    - The largest n_assets companies are selected based on end-of-year market cap
    - Tickers for the sampling year are saved
    - Returns and volatilities are saved
            for the sampling year and subsequent years.
    '''
    # select permnos
    if last_year:
        permnos = src.preprocess.n_largest_permnos(year=year, n=n_assets, consider_next=False)
        subsequent_years = 0
    else:
        permnos = src.preprocess.n_largest_permnos(year=year, n=n_assets)
    
    # filesystem management
    if not os.path.exists('../data/processed/annual/{}'.format(year)):
        os.makedirs('../data/processed/annual/{}'.format(year))
    
    for load_year in range(year, year+subsequent_years+1):
        # load & filter
        df_year = src.loader.load_crsp_year(year=load_year)
        df_filtered = src.preprocess.filter_permnos(df_year, permnos)
        
        # tickers
        if load_year == year:
            permno_to_ticker = src.preprocess.make_ticker_series(df_filtered)
            permno_to_ticker.to_csv('../data/processed/annual/{}/tickers.csv'.format(year))
            
        # returns
        returns = src.preprocess.make_return_matrix(df_filtered)
        returns.to_csv('../data/processed/annual/{}/returns_{}.csv'.format(year, load_year))
        
        #vola
        volas = src.preprocess.make_vola_matrix(df_filtered)
        volas.to_csv('../data/processed/annual/{}/volas_{}.csv'.format(year, load_year))

def preprocess_spy():
    '''Preprocesses SPY data and and stores CSV files.'''
    # load
    spy = pd.read_pickle('../data/raw/spy.pkl')
    
    # annualise vola
    spy['vola'] = (spy['vola'])# * np.sqrt(250))
    
    # save CSVs
    spy.to_csv('../data/processed/factors/spy.csv')
    spy['ret'].to_csv('../data/processed/factors/spy_ret.csv')
    spy['vola'].to_csv('../data/processed/factors/spy_vola.csv')
    
def preprocess_ff_factors():
    '''Preprocesses FF factor data and and stores CSV files.'''
    # load
    ff_factors = pd.read_pickle('../data/raw/ff_factors.pkl')
    
    # save CSVs
    ff_factors.to_csv('../data/processed/factors/ff_factors.csv')
    ff_factors['mktrf'].to_csv('../data/processed/factors/capm.csv')
    ff_factors['rf'].to_csv('../data/processed/factors/rf.csv')
    ff_factors[['mktrf', 'smb', 'hml']].to_csv('../data/processed/factors/ff3f.csv')
    ff_factors[['mktrf', 'smb', 'hml', 'umd']].to_csv('../data/processed/factors/c4f.csv')