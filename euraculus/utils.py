"""This module contains a collection of helper functions for the project."""

import numpy as np
import pandas as pd
from string import ascii_uppercase as ALPHABET


# def lookup_ticker(ticker, year):
#     """Returns descriptive data for a given ticker in a given year"""
#     data = euraculus.loader.load_descriptive().reset_index().set_index("ticker")
#     data = data[(data.namedt.dt.year <= year) & (data.nameendt.dt.year >= year)]
#     data = data.loc[ticker]
#     return data


def make_ticker_dict(tickers: list) -> dict:
    """Create column-to-ticker dictionary and add .<ALPHA> to duplicate tickers.

    Args:
        tickers: list of ticker strings, e.g ["ABC", "XYZ", "OPQ", "ABC", "KLM"]

    Returns:
        column_to_ticker: column, ticker pairs with identifying letters appended,
            ["ABC.A", "XYZ", "OPQ", "ABC.B", "KLM"]

    """
    # set up output
    column_to_ticker = {i: ticker for i, ticker in enumerate(tickers)}

    # find indices for each unique ticker
    for ticker in set(tickers):
        ticker_indices = [
            col for col, value in column_to_ticker.items() if value == ticker
        ]

        # append if duplicate
        if len(ticker_indices) > 1:
            for occurence, ticker_index in enumerate(ticker_indices):
                column_to_ticker[ticker_index] += "." + ALPHABET[occurence]

    return column_to_ticker




def matrix_asymmetry(M: np.ndarray, drop_diag: bool = False) -> float:
    """Return a (self-built) measure of matrix asymmetry.

    The measure is calculated as asymmetry = |M_a| / |M_s|, where
        |.|: Frobenius norm of .
        M_a: the asymmetric part of M = (M - M') / 2
        M_s: the symmetric part of M = (M + M') / 2
    The matrix diagonal can be excluded by setting drop_diag=True.

    Args:
        M: The (square) matrix to be analyzed.
        drop_diag: Indicates if the diagonal should be excluded from the calculations.

    Returns:
        asymmetry: The calculated symmetry measure.

    """
    M_ = M.copy()
    if drop_diag:
        M_diag = np.diag(np.diag(M))
        M_ -= M_diag
    M_s = (M_ + M_.T) / 2  # symmetric_part
    M_a = (M_ - M_.T) / 2  # asymmetric_part
    asymmetry = np.linalg.norm(M_a) / np.linalg.norm(M_s)
    return asymmetry


def shrinkage_factor(
    array: np.ndarray,
    benchmark_array: np.ndarray,
    drop_zeros: bool = False,
) -> float:
    """Calculate the average shrinkage factor of the elements in an array.

    Shrinkage is calculated with regards to a benchmark array of the same dimensions.
    Zero values in array can be excluded by setting drop_zeros=True.

    Args:
        array: The array to be evaluated.
        benchmark_array: A benchmark of the same array dimensions as array.
        drop_zeros: Indicates if zero elements should be excluded from the calculations.

    Returns:
        mean_scaling: The average scaling factor of the non-zero elements.

    """
    if drop_zeros:
        benchmark_array = benchmark_array[array != 0]
        array = array[array != 0]

    mean_shrinkage = 1 - abs(array).mean() / abs(benchmark_array).mean()

    return mean_shrinkage


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Transform a covariance matrix into a correlation matrix.

    Args:
        cov: Covariance matrix to be transformed.

    Returns:
        corr: Correlation matrix corresponding to the input covariance matrix.

    """
    stds = np.sqrt(np.diag(cov)).reshape(-1, 1)
    std_prod = stds @ stds.T
    corr = cov / std_prod
    return corr


# def calculate_nonzero_shrinkage(
#     array: np.ndarray,
#     benchmark_array: np.ndarray,
# ) -> float:
#     """Calculate the shrinkage factor of the nonzero elements in an array.

#     Shrinkage is calculated with regards to a benchmark array of the same dimensions.

#     Args:
#         array: The array to be evaluated.
#         benchmark_array: A benchmark of the same array dimensions as array.

#     Returns:
#         mean_scaling: The average scaling factor of the non-zero elements.

#     """
#     mean_scaling = (
#         1 - abs(array[array != 0]).mean() / abs(benchmark_array[array != 0]).mean()
#     )
#     return mean_scaling


# def calculate_full_shrinkage(
#     array: np.ndarray,
#     benchmark_array: np.ndarray,
# ) -> float:
#     """Calculates the scaling factor of all elements in an array.

#     Shrinkage is calculated with regards to a benchmark array of the same dimensions.

#     Args:
#         array: The array to be evaluated.
#         benchmark_array: A benchmark of the same array dimensions as array.

#     Returns:
#         mean_scaling: The average scaling factor of the non-zero elements.

#     """
#     mean_shrinkage = 1 - abs(array).mean() / abs(benchmark_array).mean()
#     return mean_shrinkage


def autocorrcoef(X, lag=1):
    """Returns the autocorrelation matrix with input number of lags."""
    N = X.shape[1]
    autocorr = np.corrcoef(X[lag:], X.shift(lag)[lag:], rowvar=False)[N:, :N]
    return autocorr


def prec_to_pcorr(prec):
    """Returns a partial correlation matrix corresponding
    to the input precision matrix.
    """
    stds = np.sqrt(np.diag(prec)).reshape(-1, 1)
    std_prod = stds @ stds.T
    pcorr = -prec / std_prod
    np.fill_diagonal(pcorr, 1)
    return pcorr


def average_correlation(data: pd.DataFrame) -> float:
    """Calculate the average correlation coefficient among the columns of an array.

    Average correlation is calculated as the average off-diagonal value in the
    Pearson correlation matrix.

    Args:
        data: The dataset to be evaluated.

    Returns:
        mean_corr: The calculated average correaltion.

    """
    n = data.shape[1]
    corr = data.corr().values
    cross_corrs = corr[np.triu_indices(n, k=1)]
    mean_corr = cross_corrs.mean()
    return mean_corr


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
    """Returns a dictionary {column_number: ticker}."""
    column_to_permno = {
        k: v for (k, v) in zip(range(df_timeseries.shape[1]), df_timeseries.columns)
    }
    permno_to_ticker = {
        str(int(k)): v for (k, v) in zip(df_descriptive.permno, df_descriptive.ticker)
    }
    column_to_ticker = {
        i: permno_to_ticker[k] for i, k in enumerate(column_to_permno.values())
    }
    return column_to_ticker


# def summarise_returns(df):
#     """Returns the total return and annualised variance for
#     an input dataframe of monthly sampled data.
#     """
#     df = df.unstack("permno")
#     df[[("weight", permno) for permno in df["mcap"].columns]] = df["mcap"] / df[
#         "mcap"
#     ].sum(axis=1).values.reshape(-1, 1)

#     # build indices
#     df_aggregate = pd.DataFrame(index=df.index)
#     df_aggregate["ew_ret"] = df["retadj"].mean(axis=1)
#     df_aggregate["vw_ret"] = (df["retadj"] * df["weight"]).sum(axis=1)
#     df_aggregate["ew_var"] = df["var"].mean(axis=1)
#     df_aggregate["vw_var"] = (df["var"] * df["weight"]).sum(axis=1)

#     df_stats = pd.DataFrame()
#     df_stats["ret"] = (1 + df["retadj"]).prod() - 1
#     df_stats.loc["ew", "ret"] = (1 + df_aggregate["ew_ret"]).prod() - 1
#     df_stats.loc["vw", "ret"] = (1 + df_aggregate["vw_ret"]).prod() - 1
#     df_stats["var"] = df["retadj"].var() * 252
#     df_stats.loc["ew", "var"] = df_aggregate["ew_ret"].var() * 252
#     df_stats.loc["vw", "var"] = df_aggregate["vw_ret"].var() * 252

#     return df_stats
