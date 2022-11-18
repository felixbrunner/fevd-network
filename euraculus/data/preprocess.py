import warnings
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from euraculus.data.map import DataMap


def map_columns(
    df: pd.DataFrame,
    mapping: pd.Series,
    mapping_name: str = None,
) -> pd.DataFrame:
    """Transform column index given a mapping.

    Args:
        df: Original DataFrame with permnos as columns.
        mapping: Series with column mapping.
        mapping_name: The name of the new mapping

    Returns:
        df_: Relabeled DataFrame with newly mapped columns.

    """
    df_ = df.rename(columns=dict(mapping))
    df_.columns.name = mapping_name

    return df_


def prepare_log_data(df_data: pd.DataFrame, df_fill: pd.DataFrame) -> pd.DataFrame:
    """Fills missing intraday data with alternative data source, then take logs.

    Args:
        df_data: Intraday observations, e.g. volatilities.
        df_fill: Alternative data, e.g. end of day bid-as spreads.

    Returns:
        df_logs: Logarithms of filled intraday variances.

    """
    # prepare filling values
    no_fill = df_fill.bfill().isna()
    minima = df_fill.replace(0, np.nan).min()
    df_fill = df_fill.replace(0, np.nan).ffill().fillna(value=minima)
    df_fill[no_fill] = np.nan

    # fill in missing data
    df_data[(df_data == 0) | (df_data.isna())] = df_fill
    minima = df_data.replace(0, np.nan).min()
    df_data = df_data.replace(0, np.nan).ffill().fillna(value=minima)

    # logarithms
    df_logs = np.log(df_data)
    return df_logs


def log_replace(df: pd.DataFrame, method: str = "min") -> pd.DataFrame:
    """Take logarithms of input DataFrame and fill missing values.

    The method argument specifies how missing values after taking logarithms
    are to be filled (includes negative values before taking logs).

    Args:
        df: Input data in a DataFrame.
        method: Method to fill missing values (includes negative values
            before taking logs). Options are ["min", "mean", "interpolate", "zero"].

    Returns:
        df_: The transformed data.

    """
    # logarithms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df_ = np.log(df)
        df_[df_ == -np.inf] = np.nan

    # fill missing
    if method == "min":
        df_ = df_.fillna(value=df_.min())
    elif method == "mean":
        df_ = df_.fillna(df_.mean())
    elif method == "interpolate":
        df_ = df_.interpolate()
    elif method == "zero":
        df_ = df_.fillna(0)
    elif method == "ffill":
        df_ = df_.ffill()
    else:
        raise ValueError("method '{}' not defined".format(method))

    # fill reamining gaps (e.g., at beginning when forward filling)
    n_missing = df_.isna().sum().sum()
    if n_missing > 0:
        warnings.warn(f"filling {n_missing} missing values with minimum")
        df_ = df_.fillna(value=df_.min())

    return df_


def construct_index(
    df: pd.DataFrame,
    column: str = "retadj",
    weighting_column: str = None,
    logs: bool = False,
) -> pd.Series:
    """Construct a (weighted) index across the included assets.

    Args:
        df: MultiIndexed DataFrame with columns 'var' and 'noisevar'.
        column: Column name to construct the index.
        weighting_column: Column name to weigh index (optional).
        logs: Indicates whether index should be build from log volatility.

    Returns:
        index: Constructed index series.
    """
    df_data = df[column].unstack()
    if logs:
        df_data = np.log(df_data.replace(0, np.nan))
    if weighting_column is not None:
        df_weights = df[weighting_column].unstack()
        df_weights = df_weights.div(df_weights.sum(axis=1), axis=0)
        index = df_data.mul(df_weights).sum(axis=1).rename("index_weighted")
    else:
        index = df_data.mean(axis=1).rename("index_ew")

    return index


def count_obs(df: pd.DataFrame, column: str = "retadj") -> pd.Series:
    """Count observations per period across the included assets.

    Args:
        df: MultiIndexed DataFrame with columns 'var' and 'noisevar'.
        column: Column name to count observations.

    Returns:
        count: Constructed count series.
    """
    df_data = df[column].unstack()
    count = df_data.count(axis=1).rename(f"count")
    return count


def construct_normalized_vola_index(df: pd.DataFrame, logs: bool = False) -> pd.Series:
    """Constructs an equally weighted normalized intraday volatility index across the included assets.

    Args:
        df: MultiIndexed DataFrame with columns 'var' and 'noisevar'.
        logs: Indicates whether index should be build from log volatility.

    Returns:
        index: Constructed index series.
    """
    df_var = df["var"].unstack()
    df_noisevar = df["noisevar"].unstack()
    df_var[df_var == 0] = df_noisevar
    df_vola = np.sqrt(df_var.replace(0, np.nan))
    if logs:
        df_vola = np.log(df_vola)
    index = (
        df_vola.sub(df_vola.mean()).div(df_vola.std()).mean(axis=1).rename("vola_index")
    )
    return index


def construct_pca_factors(df: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    """Extracts the first principal components from a dataframe.

    Args:
        df: The dataset to extract the PCs.
        n_factors: The number of components to be extracted.

    Returns:
        df_pca: Dataframe with the first PCs.

    """
    pca = PCA(n_components=n_factors)
    df_pca = pd.DataFrame(
        data=pca.fit_transform(df),
        index=df.index,
        columns=[f"pca_{i+1}" for i in range(n_factors)],
    )
    return df_pca
