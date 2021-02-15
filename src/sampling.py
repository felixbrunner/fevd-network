import numpy as np
import pandas as pd
import os
import src

#from sklearn.linear_model import LinearRegression


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
    permnos = summarise_crsp_year(year, consider_next)\
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

# OLD FUNCTION
# def preprocess_year(year, n_assets=100, subsequent_years=1, last_year=False):
#     '''Preprocesses a single sampling date in the following way:
#     - The largest n_assets companies are selected based on end-of-year market cap
#     - Tickers for the sampling year are saved
#     - Returns and volatilities are saved
#             for the sampling year and subsequent years.
#     '''
#     # select permnos
#     if last_year:
#         permnos = n_largest_permnos(year=year, n=n_assets, consider_next=False)
#         subsequent_years = 0
#     else:
#         permnos = n_largest_permnos(year=year, n=n_assets)
    
#     # filesystem management
#     if not os.path.exists('../data/processed/annual/{}'.format(year)):
#         os.mkdir('../data/processed/annual/{}'.format(year))
    
#     for load_year in range(year, year+subsequent_years+1):
#         # load & filter
#         df_year = src.loader.load_crsp_year(year=load_year)
#         df_filtered = filter_permnos(df_year, permnos)
        
#         # tickers
#         if load_year == year:
#             permno_to_ticker = make_ticker_series(df_filtered)
#             permno_to_ticker.to_csv('../data/processed/annual/{}/tickers.csv'.format(year))
            
#         # returns
#         returns = make_return_matrix(df_filtered)
#         returns.to_csv('../data/processed/annual/{}/returns_{}.csv'.format(year, load_year))
        
#         #vola
#         volas = make_vola_matrix(df_filtered)
#         volas.to_csv('../data/processed/annual/{}/volas_{}.csv'.format(year, load_year))

def preprocess_spy():
    '''Preprocesses SPY data and and stores CSV files.'''
    # load
    spy = pd.read_pickle('../data/raw/spy.pkl')
    
    # annualise var
    spy['var'] = (spy['var'])# * np.sqrt(250))
    
    # save CSVs
    spy.to_csv('../data/processed/factors/spy.csv')
    spy['ret'].to_csv('../data/processed/factors/spy_ret.csv')
    spy['var'].to_csv('../data/processed/factors/spy_var.csv')
    
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

def load_sampling_data(year, month=12, months_back=12, months_forward=12):
    '''Loads a DataFrame containing the data for the specified window.
    If forward window reaches into unavailable data, this period is ignored.
    '''
    # define parameters
    steps_back = abs((month-months_back) // 12)
    steps_forward = (month+months_forward-1) // 12
    
    # load complete dataframe
    df = pd.DataFrame()
    for y in range(year-steps_back, year+steps_forward+1):
        if y <= year:
            df = df.append(src.loader.load_crsp_year(y).sort_index())
        else:
            try:
                df = df.append(src.loader.load_crsp_year(y).sort_index())
            except:
                pass
    
    # construct backwards dataframe
    df_back = pd.DataFrame()
    y = year
    m = month
    while months_back>0:
        df_back = df[(df.index.get_level_values('date').year==y) & (df.index.get_level_values('date').month==m)].append(df_back)
        if m > 1:
            m -= 1
        else:
            m = 12
            y -= 1
        months_back -= 1
        
    # construct forward dataframe
    df_forward = pd.DataFrame()
    y = year
    m = month
    while months_forward>0:
        if m < 12:
            m += 1
        else:
            m = 1
            y += 1
        try:
            df_forward = df_forward.append(df[(df.index.get_level_values('date').year==y) & (df.index.get_level_values('date').month==m)])
        except:
            pass
        months_forward -= 1
    
    return (df_back, df_forward)

def describe_assets(df_back, df_forward):
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
    permnos = df_back.reset_index().permno.unique()

    # assemble
    df_summary = _tickers(df_back).to_frame()\
                    .join(_has_all_days(df_back))\
                    .join(_has_obs(df_forward, permnos).rename('has_next_obs'))\
                    .join(_last_size(df_back))\
                    .join(_mean_size(df_back))
    
    # set next obs to TRUE if there are no subsequent observations at all
    if not any(df_summary['has_next_obs']):
        df_summary['has_next_obs'] = True
    
    # cross-sectional size rank
    df_summary['size_rank'] = df_summary['last_size']\
                                .where(df_summary['has_all_days'])\
                                .where(df_summary['has_next_obs'])\
                                .rank(ascending=False)
    
    return df_summary

def select_permnos(df_summary, n_assets=100):
    '''Returns a list of permnos of the n tickers
    with the highest market capitalisation end of year.'''
    permnos = df_summary\
                    .sort_values('size_rank')\
                    .head(n_assets)\
                    .reset_index()\
                    .permno\
                    .values\
                    .tolist()
    return permnos

def preprocess(year, month=12, n_assets=100, months_back=12, months_forward=12):
    '''Prepares asset return and variance datasets for each (year, month) tuple
    and stores them in the specific path.
    '''
    # load data
    df_back, df_forward = load_sampling_data(year, month, months_back=months_back, months_forward=months_forward)
    
    # select assets
    df_summary = describe_assets(df_back, df_forward)
    permnos = select_permnos(df_summary, n_assets=100)
    
    # slice
    df_back = df_back[df_back.index.isin(permnos, level='permno')]
    df_forward = df_forward[df_forward.index.isin(permnos, level='permno')]
    
    # dump
    path = '../data/processed/monthly/{}/{}/'.format(year, month)
    if not os.path.exists(path):
        os.mkdir(path)
    df_back.to_csv(path+'df_back.csv')
    df_forward.to_csv(path+'df_forward.csv')
