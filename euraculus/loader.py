"""
This module provides functions to load data from disk.

"""

import pandas as pd

# import numpy as np
# import scipy as sp
# import datetime as dt

# import euraculus


# # moved to datamap.load_crsp_data
# def load_crsp_year(year: int) -> pd.DataFrame:
#     """Read a single year of raw crsp data.

#     Used in the sampling step.

#     Args:
#         year (int): Year of data to be loaded.

#     Returns:
#         df (pandas.DataFrame): Loaded data in tabular form.
#     """
#     df = pd.read_pickle("../data/raw/crsp_{}.pkl".format(year))
#     return df


# def load_spy(year: int = None, columns: list = None) -> pd.DataFrame:
#     """Loads SPY data from disk.

#     This is used in factor model step.

#     Args:
#         year (int): Year to be loaded (optional), defaults to loading all years.
#         columns (list): Column names to be included in output.

#     Returns:
#         df (pandas.DataFrame): Selected SPY data in tabular format.

#     """
#     df = pd.read_pickle("../data/raw/spy.pkl").reset_index()
#     df["date"] = pd.to_datetime(df["date"], yearfirst=True)
#     df = df.set_index("date")
#     if year:
#         df = df[df.index.year == year]
#     if columns:
#         df = df[columns]
#     return df


# def load_factors(year: int = None) -> pd.DataFrame:
#     """Load factor data from disk.

#     Returns factor data as pandas DataFrame.
#     Can be limited to a single year.

#     Args:
#         year (int): Year to be loaded (optional), defaults to loading all years.

#     Returns:
#         df (pandas.DataFrame): Factor data in tabular format.

#     """
#     df = pd.read_pickle("../data/raw/ff_factors.pkl").reset_index()
#     df["date"] = pd.to_datetime(df["date"], yearfirst=True)
#     df = df.set_index("date")
#     if year is not None:
#         df = df[df.index.year == year]
#     return df


def load_monthly_crsp(year: int, month: int, which: str = "back", column: str = None):
    """Loads monthly sampled CRSP data from disk.

    Args:
        year (int): Year of the sampling date.
        month (int): Month of the sampling date.
        which (str): Defines if forward or backward looking data should be loaded.
            Options are 'back' and 'forward'.
        column (str): Name of a single column to be loaded (optional).

    Returns:
        df (pandas.DataFrame): CRSP data in tabular form.
    """
    # load raw
    if which == "back":
        df = pd.read_csv(
            "../data/processed/monthly/{}/{}/df_back.csv".format(year, month)
        )
    elif which == "forward":
        df = pd.read_csv(
            "../data/processed/monthly/{}/{}/df_forward.csv".format(year, month)
        )

    # format
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "permno"])

    # return data matrix if column is chosen
    if column:
        df = df[column].unstack()

    return df.sort_index()


def load_monthly_estimation_data(year, month, column=None):
    """Loads a merged dataframe with daily observations.
    'year' and 'month' define the sampling date.
    'column' argument (optional) can be specified to load a single column as data matrix.
    """
    # load raw
    df_back = pd.read_csv(
        "../data/processed/monthly/{}/{}/df_back.csv".format(year, month)
    )
    df_back_residuals = pd.read_csv(
        "../data/processed/monthly/{}/{}/df_back_residuals.csv".format(year, month)
    )
    df_var_decomposition = pd.read_csv(
        "../data/processed/monthly/{}/{}/df_var_decomposition.csv".format(year, month)
    )

    # merge
    df = df_back.merge(df_back_residuals, on=["date", "permno"], how="outer").merge(
        df_var_decomposition, on=["date", "permno"], how="outer"
    )

    # format
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "permno"])

    # return data matrix if column is chosen
    if column:
        df = df[column].unstack()

    return df.sort_index()


def load_year(sampling_year, data="returns", data_year=None):
    """Returns the data for a specific year.
    Can be either estimation or analysis data (next year).
    """
    if not data_year:
        data_year = sampling_year
    try:
        df = pd.read_csv(
            "../data/processed/annual/{}/{}_{}.csv".format(
                sampling_year, data, data_year
            )
        )
    except:
        raise Exception("Data type <{}> could not be found.".format(data))
    df["date"] = pd.to_datetime(df["date"], yearfirst=True)
    df = df.set_index("date")
    return df


# def load_spy_returns(year=None):
#     '''Returns annual SPY log vola data as pandas DataFrame.
#     Can be limited to a single year.
#     '''
#     df = pd.read_csv('../data/processed/factors/spy_ret.csv')
#     df['date'] = pd.to_datetime(df['date'], yearfirst=True)
#     df = df.set_index('date')
#     if year is not None:
#         df = df[df.index.year == year]
#     return df

# def load_spy_vola(year=None):
#     '''Returns annual SPY log vola data as pandas DataFrame.
#     Can be limited to a single year.
#     '''
#     df = pd.read_csv('../data/processed/factors/spy_vola.csv')
#     df['date'] = pd.to_datetime(df['date'], yearfirst=True)
#     df = df.set_index('date')
#     if year is not None:
#         df = df[df.index.year == year]
#     return df


# replaced by DataMap.load_descriptive_data
def load_year_tickers(year):
    """Returns a dictionary"""
    tickers = pd.read_csv("../data/processed/annual/{}/tickers.csv".format(year))
    tickers = dict(tickers.set_index("permno"))
    return tickers


# replaced by DataMap.load_descriptive_data
# def load_descriptive():
#     """Loads the descriptive data and returns pandas DataFrame."""
#     df_desc = pd.read_csv("../data/processed/descriptive.csv")
#     df_desc = df_desc.astype({"permno": int, "exchcd": int}).set_index("permno")
#     df_desc[["namedt", "nameendt"]] = df_desc[["namedt", "nameendt"]].apply(
#         pd.to_datetime, format="%Y-%m-%d"
#     )
#     return df_desc


def load_rf(year=None, month=None, which="back"):
    """Returns risk-free rate data as pandas DataFrame.
    Can be limited to a single year, month.
    Set which parameter to 'forward' to obtain following period.
    """
    df = pd.read_csv("../data/processed/factors/rf.csv")
    df["date"] = pd.to_datetime(df["date"], yearfirst=True)
    df = df.set_index("date")
    if year is not None:
        if month is None:
            month = 12
        select = pd.period_range(periods=12, end=str(year) + "-" + str(month), freq="M")
        if which == "forward":
            select += 12
        df = df[df.index.to_period("M").isin(select)]
    return df.sort_index()


# def load_descriptive():
#     '''Loads the descriptive data and returns pandas DataFrame.'''
#     df_desc = pd.read_pickle('../data/raw/df_crsp_desc.pkl')
#     df_desc = df_desc.astype({'permno': int, 'exchcd':int}) \
#                      .set_index('permno')
#     df_desc[['st_date','end_date']] = df_desc[['st_date','end_date']].apply(pd.to_datetime, format='%Y-%m-%d')
#     return df_desc


# def load_year_all(year):
#     '''Returns the estimation and analysis data for a specific year.'''
#     df_est = load_year(year, purpose='estimation')
#     df_ana = load_year(year, purpose='analysis')
#     df_factors_est = load_factors(year=year)
#     df_factors_ana = load_factors(year=year+1)
#     return (df_est, df_ana, df_factors_est, df_factors_ana)

# def select_factor_data(df_factors_est, df_factors_ana, selected_factors=[]):
#     '''Returns a subset of factors as a pandas DataFrame.'''
#     df_est = df_factors_est[selected_factors]
#     df_ana = df_factors_ana[selected_factors]
#     return (df_est, df_ana)


def load_year_vola(year, purpose="est"):
    """Returns the data for a specific year.
    Can be either estimation or analysis data (next year).
    """
    assert purpose in [
        "est",
        "ana",
    ], "purpose needs to be either estimation or analysis"
    df = pd.read_csv(
        "../data/processed/yearly/vola/df_{}_{}.csv".format(purpose, str(year))
    )
    df["date"] = pd.to_datetime(df["date"], yearfirst=True)
    df = df.set_index("date")
    return df


def load_year_all_vola(year):
    """Returns the estimation and analysis data for a specific year."""
    df_est = load_year_vola(year, purpose="est")
    df_ana = load_year_vola(year, purpose="ana")
    df_factors_est = load_factors(year=year)
    df_factors_ana = load_factors(year=year + 1)
    return (df_est, df_ana, df_factors_est, df_factors_ana)


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
