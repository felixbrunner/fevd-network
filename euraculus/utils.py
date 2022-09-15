"""This module contains a collection of helper functions for the project."""

import datetime as dt
import numpy as np
import pandas as pd


def matrix_asymmetry(M: np.ndarray, drop_diag: bool = False) -> float:
    """Return a (self-built) measure of matrix asymmetry.

    The measure is calculated as asymmetry = |M_a| / |M|, where
        |.|: The entry-wise L1 norm of the matrix, |M| = |vec(M)|.
        M_a: the asymmetric part of M = (M - M') / 2
             (the symmetric part of M = (M + M') / 2)
    The matrix diagonal can be excluded by setting drop_diag=True.

    Args:
        M: The (square) matrix to be analyzed.
        drop_diag: Indicates if the diagonal should be excluded from the calculations.

    Returns:
        asymmetry: The calculated asymmetry measure.
    """
    M_ = M.copy()
    if drop_diag:
        M_diag = np.diag(np.diag(M))
        M_ -= M_diag
    # M_s = (M_ + M_.T) / 2  # symmetric_part
    M_a = (M_ - M_.T) / 2  # asymmetric_part
    asymmetry = abs(M_a).sum() / abs(M_).sum()
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


def autocorrcoef(X, lag=1):
    """Returns the autocorrelation matrix with input number of lags."""
    N = X.shape[1]
    autocorr = np.corrcoef(X[lag:], X.shift(lag)[lag:], rowvar=False)[N:, :N]
    if type(X) == pd.DataFrame:
        autocorr = pd.DataFrame(data=autocorr, index=X.columns, columns=X.columns)
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


def months_difference(start_date: dt.datetime, end_date: dt.datetime) -> int:
    """Get the difference of two datetime objects in calendar months.

    Args:
        start_date: Earlier datetime.
        end_date: Later datetime

    Returns:
        delta: Difference of dates in full months.
    """
    delta = 12 * (end_date.year - start_date.year) + (end_date.month - start_date.month)
    return delta


def herfindahl_index(values: np.ndarray, axis: int = 0) -> float:
    """Calculate the Herfindahl-Hirschman index of concentration over an array of values.

    Args:
        values: Array containing the values.
        axis: The axis along which to calculate the index.

    Returns:
        hhi: The Herfindahl-Hirschman index of concentration.
    """
    weights = values / values.sum(axis=axis, keepdims=True)
    hhi = (weights**2).sum(axis=axis)
    if hhi.size == 1:
        hhi = hhi[0]
    return hhi


def power_law_exponent(
    sample: np.ndarray,
    axis: int = 0,
    invert: bool = False,
) -> float:
    """Calculate the exponent in a Pareto distributed sample.

    Args:
        sample: Array of values making up the sample.
        axis: The axis along which to calculate the exponent.
        invert: Indicates if exponent should be inverted to measure concentration.

    Returns:
        alpha: The exponent of the power law distribution.
    """
    # keep only positive values
    sample[sample <= 0] = np.nan

    # calculate
    n = (~np.isnan(sample)).sum(axis=axis, keepdims=True)
    v_min = np.nanmin(sample, axis=axis, keepdims=True)
    alpha = 1 + n * np.nansum(np.log(sample / v_min), axis=axis, keepdims=True) ** -1
    if invert:
        alpha **= -1
    if alpha.size == 1:
        alpha = alpha.flat[0]
    return alpha
