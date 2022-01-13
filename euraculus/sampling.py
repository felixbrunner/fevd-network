"""
This module provides a set uf functions to sample the largest companies from
CRSP data at the end of each month.

"""

import pandas as pd
import os
import euraculus


def preprocess_ff_factors():
    """Preprocess FF factor data and and stores CSV files."""
    # load
    ff_factors = pd.read_pickle("../data/raw/ff_factors.pkl")

    # save CSVs
    ff_factors.to_csv("../data/processed/factors/ff_factors.csv")
    ff_factors["mktrf"].to_csv("../data/processed/factors/capm.csv")
    ff_factors["rf"].to_csv("../data/processed/factors/rf.csv")
    ff_factors[["mktrf", "smb", "hml"]].to_csv("../data/processed/factors/ff3f.csv")
    ff_factors[["mktrf", "smb", "hml", "umd"]].to_csv(
        "../data/processed/factors/c4f.csv"
    )


def preprocess_spy():
    """Preprocess SPY data and and stores CSV files."""
    # load
    spy = pd.read_pickle("../data/raw/spy.pkl")

    # annualise var
    spy["var"] = spy["var"]  # * np.sqrt(250))

    # save CSVs
    spy.to_csv("../data/processed/factors/spy.csv")
    spy["ret"].to_csv("../data/processed/factors/spy_ret.csv")
    spy["var"].to_csv("../data/processed/factors/spy_var.csv")


def load_sampling_data(
    year: int, month: int = 12, months_back: int = 12, months_forward: int = 12
):
    """Extract backward and forward looking data given a sampling date.

    Load a DataFrame containing the data for the specified window.
    If forward window reaches into unavailable data, this period is ignored.

    Args:
        year (int): Specifies the year of the sampling date.
        month (int): Specifies the month of the sampling date.
        months_back (int): Number of previous months to include.
        months_forward (int): Number of subsequent months to include.

    Returns:
        df_back (pandas.DataFrame): Data of the prevous months.
        df_forward (pandas.DataFrame): Data of the subsequent months.

    """
    # define parameters
    steps_back = abs((month - months_back) // 12)
    steps_forward = (month + months_forward - 1) // 12

    # load complete dataframe
    df = pd.DataFrame()
    for y in range(year - steps_back, year + steps_forward + 1):
        if y <= year:
            df = df.append(euraculus.loader.load_crsp_year(y).sort_index())
        else:
            try:
                df = df.append(euraculus.loader.load_crsp_year(y).sort_index())
            except:
                pass

    # construct backwards dataframe
    df_back = pd.DataFrame()
    y = year
    m = month
    while months_back > 0:
        df_back = df[
            (df.index.get_level_values("date").year == y)
            & (df.index.get_level_values("date").month == m)
        ].append(df_back)
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
    while months_forward > 0:
        if m < 12:
            m += 1
        else:
            m = 1
            y += 1
        try:
            df_forward = df_forward.append(
                df[
                    (df.index.get_level_values("date").year == y)
                    & (df.index.get_level_values("date").month == m)
                ]
            )
        except:
            pass
        months_forward -= 1

    return (df_back, df_forward)


def has_all_days(df: pd.DataFrame) -> pd.Series:
    """Check data completeness.

    Return a Series of booleans to indicate whether permnos
    have more than 99% of days in dataframe.

    Args:
        df (pandas.DataFrame): Financial return data with column 'retadj'.

    Returns:
        has_obs (pandas.Series): Permno, bool pairs indicating completeness.

    """
    t_periods = df.reset_index().date.nunique()
    has_all_days = df["retadj"].groupby("permno").count() > t_periods * 0.99
    has_all_days = has_all_days.rename("has_all_days")
    return has_all_days


def has_obs(df: pd.DataFrame, permnos: list) -> pd.Series:
    """Check for observations given some permnos.

    Return a Series of booleans to indicate whether there are
    observations in dataframe given an iterable of permnos.

    Args:
        df (pandas.DataFrame): Financial return data.
        permnos (list): Selection of permnos in a list.

    Returns:
        has_obs (pandas.Series): Permno, bool pairs indicating completeness.

    """
    df = df.copy().reset_index()
    permnos_in_df = df.permno.unique()
    has_obs = [permno in permnos_in_df for permno in permnos]
    has_obs = pd.Series(has_obs, index=permnos, name="has_obs")
    return has_obs


def get_last_sizes(df: pd.DataFrame) -> pd.Series:
    """Get last firm sizes from dataset.

    Create a series of last observed market capitalisation for
    all contained permnos.

    Args:
        df (pandas.DataFrame): Dataset with column 'mcap'.

    Returns:
        last_size (pandas.Series): Permno, size pairs with last observed sizes.

    """
    last_size = (
        df["mcap"]
        .unstack()
        .sort_index()
        .fillna(method="ffill", limit=1)
        .tail(1)
        .squeeze()
        .rename("last_size")
    )
    return last_size


def get_mean_sizes(df: pd.DataFrame) -> pd.Series:
    """Get average firm sizes from dataset.

    Return a series of average market capitalisations for contained permnos.

    Args:
        df (pandas.DataFrame): Dataset with column 'mcap'.

    Returns:
        mean_size (pandas.Series): Permno, size pairs with average observed sizes.

    """
    mean_size = df["mcap"].unstack().mean().squeeze().rename("mean_size")
    return mean_size


def get_tickers(df: pd.DataFrame) -> pd.Series:
    """Get tickers for the data contained in dataframe.

    Return a series of tickers for all permos in the dataframe.

    Args:
        df (pandas.DataFrame): Dataset with column 'ticker'.

    Returns:
        tickers (pandas.Series): Permno, ticker pairs from dataframe.

    """
    tickers = df["ticker"].unstack().tail(1).squeeze().rename("ticker")
    return tickers


def describe_assets(df_back: pd.DataFrame, df_forward: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for the financial data provided.

    Return a summaring dataframe where the index consists
    of the permnos present in the dataframe.
    Columns are:
        ticker
        has_all_days
        has_obs
        last_size
        mean_size
        size_rank

    Args:
        df_back (pandas.DataFrame): Data from the previous periods.
        df_forward (pandas.DataFrame): Data from the subsequent periods.

    Returns:
        df_summary (pandas.DataFrame): Summarizing information in tabular form.
    """
    # setup
    permnos = df_back.reset_index().permno.unique()

    # assemble
    df_summary = (
        get_tickers(df_back)
        .to_frame()
        .join(has_all_days(df_back))
        .join(has_obs(df_forward, permnos).rename("has_next_obs"))
        .join(get_last_sizes(df_back))
        .join(get_mean_sizes(df_back))
    )

    # set next obs to TRUE if there are no subsequent observations at all
    if not any(df_summary["has_next_obs"]):
        df_summary["has_next_obs"] = True

    # cross-sectional size rank
    df_summary["size_rank"] = (
        df_summary["last_size"]
        .where(df_summary["has_all_days"])
        .where(df_summary["has_next_obs"])
        .rank(ascending=False)
    )

    return df_summary


def select_permnos(df_summary: pd.DataFrame, n_assets: int = 100) -> list:
    """Pick N lagest companies.

    Return a list of permnos of the n companies
    with the highest market capitalisation end of year.

    Args:
        df_summary (pandas.DataFrame): Summary data to determine selection.
        n_assets (int): Number of companies to select.

    Returns:
        permnos (list): Permnos of the N largest companies.

    """
    permnos = (
        df_summary.sort_values("size_rank")
        .head(n_assets)
        .reset_index()
        .permno.values.tolist()
    )
    return permnos


def preprocess(
    year: int,
    month: int = 12,
    n_assets: int = 100,
    months_back: int = 12,
    months_forward: int = 12,
):
    """Preprocess return data to provide sample data for estimation.

    Prepare asset return and variance datasets for each (year, month) tuple
    and stores them in the specific path.

    Args:
        year (int): Specifies the year of the sampling date.
        month (int): Specifies the month of the sampling date.
        n_assets (int): Number of companies to select.
        months_back (int): Number of previous months to include.
        months_forward (int): Number of subsequent months to include.

    """
    # load data
    df_back, df_forward = load_sampling_data(
        year, month, months_back=months_back, months_forward=months_forward
    )

    # select assets
    df_summary = describe_assets(df_back, df_forward)
    permnos = select_permnos(df_summary, n_assets=n_assets)

    # slice
    df_back = df_back[df_back.index.isin(permnos, level="permno")]
    df_forward = df_forward[df_forward.index.isin(permnos, level="permno")]

    # dump
    path = "../data/processed/monthly/{}/{}/".format(year, month)
    if not os.path.exists(path):
        os.mkdir(path)
    df_back.to_csv(path + "df_back.csv")
    df_forward.to_csv(path + "df_forward.csv")


# def summarise_crsp_year(year: int, consider_next: bool = True) -> pd.DataFrame:
#     """Create summary statistics for the financial data provided.

#     Returns a summaring dataframe where the index consists
#     of the permnos present in the dataframe.
#     Columns are:
#         ticker
#         has_all_days
#         has_obs
#         last_size
#         mean_size
#         size_rank

#     Args:
#         year (int): Year to calculate statistics
#         consider_next (bool): Indicates if following year should be considered.

#     Returns:
#         df_summary (pandas.DataFrame): Summary statistic in tabular form.

#     """
#     # setup
#     df_this = euraculus.loader.load_crsp_year(year).sort_index()
#     if consider_next:
#         df_next = euraculus.loader.load_crsp_year(year + 1).sort_index()
#     else:
#         df_next = df_this
#     permnos = df_this.reset_index().permno.unique()

#     # assemble
#     df_summary = (
#         get_tickers(df_this)
#         .to_frame()
#         .join(has_all_days(df_this))
#         .join(has_obs(df_next, permnos).rename("has_next_obs"))
#         .join(get_last_sizes(df_this))
#         .join(get_mean_sizes(df_this))
#     )

#     # cross-sectional size rank
#     df_summary["size_rank"] = (
#         df_summary["last_size"]
#         .where(df_summary["has_all_days"])
#         .where(df_summary["has_next_obs"])
#         .rank(ascending=False)
#     )

#     return df_summary


# def n_largest_permnos(year: int, n: int = 100, consider_next: bool = True) -> list:
#     """Pick N lagest companies.

#     Return a list of permnos of the n companies
#     with the highest market capitalisation end of year.

#     Args:
#         year (int): Year to carry out selection.
#         n (int): Number of companies to select.
#         consider_next (bool): Indicates if following year should be considered.

#     Returns:
#         permno (list): Permnos of the N largest companies.

#     """
#     permnos = (
#         summarise_crsp_year(year, consider_next)
#         .sort_values("size_rank")
#         .head(n)
#         .reset_index()
#         .permno.values.tolist()
#     )
#     return permnos


# def filter_permnos(df, permnos):
#     """Returns a dataframe that only contains the input permnos."""
#     df = df[df.index.isin(permnos, level="permno")]
#     return df


# def make_return_matrix(df):
#     """Returns a dataframe with daily returns.
#     Index are dates, columns are permnos.
#     """
#     df_returns = df["retadj"].unstack()
#     return df_returns


# def make_vola_matrix(df):
#     """Returns a dataframe with intraday volatility.
#     Index are dates, columns are permnos.
#     """
#     df_vola = df["vola"].unstack().fillna(0)
#     df_vola = df_vola.replace(0, 1e-8)
#     return df_vola


# def make_ticker_series(df):
#     """Returns a ticker series indexed by permno."""
#     permno_to_ticker = (
#         df["ticker"].unstack().tail(1).squeeze().rename("permno_to_ticker")
#     )
#     return permno_to_ticker
