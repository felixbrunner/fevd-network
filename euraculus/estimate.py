import warnings
from dateutil.relativedelta import relativedelta
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from euraculus.data import DataMap
from euraculus.var import FactorVAR
from euraculus.covar import GLASSO
from euraculus.fevd import FEVD
from euraculus.utils import (
    matrix_asymmetry,
    shrinkage_factor,
    herfindahl_index,
    power_law_exponent,
    months_difference,
)
import datetime as dt

import euraculus


def map_columns(
    df: pd.DataFrame,
    mapping: pd.Series,
    mapping_name: str = None,
) -> pd.DataFrame:
    """Transform column index from given a mapping.

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
    """Take logarithms of input DataFrame and fills missing values.

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


def construct_crsp_index(sampling_date: dt.datetime, data: DataMap) -> pd.Series:
    """Constructs an equally weighted log mcap volatility index across the CRSP universe.

    Args:
        data: DataMap to load data from.
        sampling_date: Last day in the sample.

    Returns:
        index: Constructed index series.

    """
    # set parameters
    start_date = sampling_date - relativedelta(years=1) + relativedelta(days=1)

    # load data
    df_var = data.load_crsp_data(
        start_date=start_date, end_date=sampling_date, column="var"
    )
    df_noisevar = data.load_crsp_data(
        start_date=start_date, end_date=sampling_date, column="noisevar"
    )
    df_ret = data.load_crsp_data(
        start_date=start_date, end_date=sampling_date, column="retadj"
    )
    df_mcap = data.load_crsp_data(
        start_date=start_date, end_date=sampling_date, column="mcap"
    )

    # process data
    df_var[df_var == 0] = df_noisevar
    df_vola = np.sqrt(df_var.replace(0, np.nan))
    df_lagged_mcap = df_mcap / (df_ret + 1)
    df_lagged_mcap[df_lagged_mcap <= 0] = np.nan
    df_log_mcap_vola = np.log(df_vola) + np.log(df_lagged_mcap)

    # build index
    index = (
        df_log_mcap_vola.sub(df_log_mcap_vola.mean())
        .div(df_log_mcap_vola.std())
        .mean(axis=1)
        .rename("crsp")
    )
    return index


def load_estimation_data(data: DataMap, sampling_date: dt.datetime) -> dict:
    """Load the data necessary for estimation from disk.

    Args:
        data: DataMap to load data from.
        sampling_date: Last day in the sample.

    Returns:
        df_info: Summarizing information.
        df_log_mcap_vola: Logarithm of value variance variable.
        df_factors: Factor data.

    """
    # asset data
    df_var = data.load_historic(sampling_date=sampling_date, column="var")
    df_noisevar = data.load_historic(sampling_date=sampling_date, column="noisevar")
    df_ret = data.load_historic(sampling_date=sampling_date, column="retadj")
    df_mcap = data.load_historic(sampling_date=sampling_date, column="mcap")
    df_info = data.load_asset_estimates(
        sampling_date=sampling_date,
        columns=[
            "ticker",
            "comnam",
            "last_mcap",
            "mean_mcap",
            "last_mcap_volatility",
            "mean_mcap_volatility",
            "sic_division",
            "ff_sector",
            "ff_sector_ticker",
            "gics_sector",
        ],
    )

    # prepare asset data
    df_vola = np.sqrt(df_var)
    df_noisevola = np.sqrt(df_noisevar)
    df_lagged_mcap = df_mcap / (df_ret + 1)
    df_log_vola = prepare_log_data(df_data=df_vola, df_fill=df_noisevola)
    df_log_mcap = log_replace(df=df_lagged_mcap, method="ffill")
    df_log_mcap_vola = df_log_vola + df_log_mcap
    df_log_mcap_vola = map_columns(
        df_log_mcap_vola, mapping=df_info["ticker"], mapping_name="ticker"
    )

    # factor data
    df_factors = pd.DataFrame(index=df_var.index)

    def prepare_spy_factor(df_spy):
        open_prc = df_spy["prc"] / (1 + df_spy["ret"])
        std = df_spy["var"] ** 0.5
        factor = log_replace(open_prc * std, method="min").rename("spy")
        return factor

    def prepare_yahoo_factor(df_yahoo):
        open_prc = df_yahoo["Open"]
        std = np.sqrt(0.3607) * (np.log(df_yahoo["High"]) - np.log(df_yahoo["Low"]))
        factor = log_replace(open_prc * std, method="min").rename("yahoo")
        return factor

    def prepare_ew_factor(df_obs):
        factor = df_obs.sub(df_obs.mean()).div(df_obs.std()).mean(axis=1).rename("ew")
        return factor

    df_spy = data.load_spy_data().reindex(df_var.index)
    spy_factor = prepare_spy_factor(df_spy)
    df_factors = df_factors.join(spy_factor)

    ew_factor = prepare_ew_factor(df_log_vola)  ##
    df_factors = df_factors.join(ew_factor)

    crsp_factor = construct_crsp_index(sampling_date=sampling_date, data=data)
    df_factors = df_factors.join(crsp_factor)

    for ticker in ["^VIX", "DX-Y.NYB", "^TNX"]:
        df_yahoo = data.load_yahoo(ticker).reindex(df_var.index)
        factor = prepare_yahoo_factor(df_yahoo).rename(ticker)
        df_factors = df_factors.join(factor)

    return (df_info, df_log_vola, df_factors)  ##


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


def build_lookup_table(df_info: pd.DataFrame) -> pd.DataFrame:
    """Build a lookup table for company names from tickers.

    Args:
        df_info: Dataframe that contains columns 'ticker' and 'comnam'.

    Returns:
        df_lookup: Lookup table with tickers as index and company names as values.

    """
    column_dict = {
        "ticker": "Tickers",
        "comnam": "Company Name",
    }
    df_lookup = (
        df_info[["ticker", "comnam"]]
        .rename(columns=column_dict)
        .set_index("Tickers")
        .sort_index()
    )
    df_lookup["Company Name"] = (
        df_lookup["Company Name"].str.title().str.replace("&", "\&")
    )

    return df_lookup


def estimate_fevd(
    var_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    var_grid: dict,
    cov_grid: dict,
) -> tuple:
    """Perform all estimation steps necessary to construct FEVD.

    Args:
        var_data: Dataframe with the data panel for the VAR.
        factor_data: Dataframe with the control factor data.
        var_grid: Grid with VAR hyperparameters.
        cov_grid: Grid with covariance hyperparameters.

    Returns:
        var_cv: Cross-validation object for VAR.
        var: The estimated VAR object.
        cov_cv: Cross-validation object for the covariance.
        cov: The estimated covariance object.
        fevd: The constructed FEVD from the estimates.

    """

    # estimate var
    var = FactorVAR(has_intercepts=True, p_lags=1)
    var_cv = var.fit_adaptive_elastic_net_cv(
        var_data=var_data,
        factor_data=factor_data,
        grid=var_grid,
        return_cv=True,
        penalize_factors=False,
    )
    residuals = var.residuals(var_data=var_data, factor_data=factor_data)

    # estimate covariance
    cov_cv = GridSearchCV(
        GLASSO(max_iter=200),
        param_grid=cov_grid,
        cv=12,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    ).fit(residuals)
    cov = cov_cv.best_estimator_

    # create fevd
    fevd = FEVD(var.var_1_matrix_, cov.covariance_)

    return (var_cv, var, cov_cv, cov, fevd)


def describe_data(df: pd.DataFrame) -> dict:
    """Creates descriptive statistics of a dataset.

    Calculates the following statistics:
        T: The number of distinct time observations.
        N: The number of distingt entities.
        nobs: The number of total available observations.

    Args:
        df: Data to be described.

    Returns:
        stats: Key, value pairs of the calculated statistics.

    """
    stats = {
        "T": df.shape[0],
        "N": df.shape[1],
        "nobs": df.notna().sum().sum(),
    }
    return stats


def describe_var(
    var: euraculus.var.VAR,
    var_cv: GridSearchCV,
    var_data: pd.DataFrame,
    factor_data: pd.DataFrame = None,
) -> dict:
    """Creates descriptive statistics of a VAR estimation.

    Calculates the following statistics:
        lambda: Overall regularization parameter.
        kappa: L1 penalty weight.
        ini_lambda: First step overall regularization parameter.
        ini_kappa: First step L1 penalty weight.
        var_matrix_density: Share of non-zero values in VAR(1) matrix.
        var_mean_connection: Average value in VAR(1) matrix.
        var_mean_abs_connection: Average absolute value in VAR(1) matrix.
        var_asymmetry: Matrix asymmetry of VAR(1) matrix.
        var_r2: Goodness of fit for full VAR model.
        var_r2_ols: Goodness of fit for full VAR model estimated by OLS.
        var_factor_r2: Goodness of fit for factors in the model.
        var_factor_r2_ols: Goodness of fit for factors in the model estimated by OLS.
        var_df_used: Total number of degrees of freedom used in VAR.
        var_nonzero_shrinkage: Average shrinkage factor of nonzero estimates.
        var_full_shrinkage: Average shrinkage factor of all estimates.
        var_factor_shrinkage: Average shrinkage factor of nonzero factor estimates.
        var_full_factor_shrinkage: Average shrinkage factor of all factor estimates.
        var_cv_loss: Average validation loss.
        var_train_loss: Average training loss.
        var_partial_r2: Partial R2 of parts of the model.
        var_component_r2: R2 of single components of the model.

    Args:
        var: Vector autoregression to be described.
        var_cv: Cross-validation from the preceeding estimation.
        var_data: Data the estimation is performed on.
        factor_data: Factor data in the estimation.

    Returns:
        stats: Key, value pairs of the calculated statistics.

    """
    ols_var = var.copy()
    ols_var.fit_ols(var_data=var_data, factor_data=factor_data)

    stats = {
        "lambda": var_cv.best_params_["lambdau"],
        "kappa": var_cv.best_params_["alpha"],
        "ini_lambda": var_cv.best_estimator_.ini_lambdau,
        "ini_kappa": var_cv.best_estimator_.ini_alpha,
        "var_matrix_density": (var.var_1_matrix_ != 0).sum() / var.var_1_matrix_.size,
        "var_mean_connection": var.var_1_matrix_.mean(),
        "var_mean_abs_connection": abs(var.var_1_matrix_).mean(),
        "var_asymmetry": matrix_asymmetry(M=var.var_1_matrix_),
        "var_r2": var.r2(var_data=var_data, factor_data=factor_data),
        "var_r2_ols": ols_var.r2(var_data=var_data, factor_data=factor_data),
        "var_factor_r2": var.factor_r2(var_data=var_data, factor_data=factor_data),
        "var_factor_r2_ols": ols_var.factor_r2(
            var_data=var_data, factor_data=factor_data
        ),
        "var_df_used": var.df_used_,
        "var_nonzero_shrinkage": shrinkage_factor(
            array=var.var_1_matrix_,
            benchmark_array=ols_var.var_1_matrix_,
            drop_zeros=True,
        ),
        "var_full_shrinkage": shrinkage_factor(
            array=var.var_1_matrix_,
            benchmark_array=ols_var.var_1_matrix_,
            drop_zeros=False,
        ),
        "var_factor_shrinkage": shrinkage_factor(
            array=var.factor_loadings_,
            benchmark_array=ols_var.factor_loadings_,
            drop_zeros=True,
        ),
        "var_full_factor_shrinkage": shrinkage_factor(
            array=var.factor_loadings_,
            benchmark_array=ols_var.factor_loadings_,
            drop_zeros=False,
        ),
        "var_cv_loss": -var_cv.best_score_,
        "var_train_loss": -var_cv.cv_results_["mean_train_score"][var_cv.best_index_],
    }

    partial_r2s = var.partial_r2s(
        var_data=var_data,
        factor_data=factor_data,
        weighting="equal",
    )
    stats.update({"var_partial_r2_" + k: v for (k, v) in partial_r2s.items()})
    component_r2s = var.component_r2s(
        var_data=var_data,
        factor_data=factor_data,
        weighting="equal",
    )
    stats.update({"var_component_r2_" + k: v for (k, v) in component_r2s.items()})

    return stats


def describe_cov(
    cov: euraculus.covar.GLASSO,
    cov_cv: GridSearchCV,
    data: pd.DataFrame,
) -> dict:
    """Creates descriptive statistics of a GLASSO Covariance estimation.

    Calculates the following statistics:
        rho: Regularization parameter.
        cov_mean_likelihood: Log-likelihood with the estimates.
        cov_mean_likelihood_sample_estimate: Log-likelihood with sample estimate.
        covar_density: Share of non-zero values in covariance matrix.
        precision_density: Share of non-zero values in precision matrix.
        covar_nonzero_shrinkage: Average shrinkage factor of nonzero covariance estimates.
        covar_full_shrinkage: Average shrinkage factor of all covariance estimates.
        precision_nonzero_shrinkage: Average shrinkage factor of nonzero precision estimates.
        precision_full_shrinkage: Average shrinkage factor of all precision estimates.
        covar_cv_loss: Average validation loss.
        covar_train_loss: Average training loss.

    Args:
        cov: GLASSO covariance estimate to be described.
        cov_cv: Cross-validation from the preceeding estimation.
        data: Data the estimation is performed on.

    Returns:
        stats: Key, value pairs of the calculated statistics.

    """
    stats = {
        "rho": cov_cv.best_params_["alpha"],
        "cov_mean_likelihood": sp.stats.multivariate_normal(cov=cov.covariance_)
        .logpdf(data)
        .mean(),  # /data.shape[1],
        "cov_mean_likelihood_sample_estimate": sp.stats.multivariate_normal(
            cov=data.cov().values
        )
        .logpdf(data)
        .mean(),  # /data.shape[1],
        "covar_density": cov.covariance_density_,
        "precision_density": cov.precision_density_,
        "covar_nonzero_shrinkage": shrinkage_factor(
            array=cov.covariance_,
            benchmark_array=data.cov().values,
            drop_zeros=True,
        ),
        "covar_full_shrinkage": shrinkage_factor(
            array=cov.covariance_,
            benchmark_array=data.cov().values,
            drop_zeros=False,
        ),
        "precision_nonzero_shrinkage": shrinkage_factor(
            array=cov.precision_,
            benchmark_array=np.linalg.inv(data.cov().values),
            drop_zeros=True,
        ),
        "precision_full_shrinkage": shrinkage_factor(
            array=cov.precision_,
            benchmark_array=np.linalg.inv(data.cov().values),
            drop_zeros=True,
        ),
        "covar_cv_loss": -cov_cv.best_score_,
        "covar_train_loss": -cov_cv.cv_results_["mean_train_score"][cov_cv.best_index_],
    }
    return stats


def describe_network(
    network: euraculus.network.Network,
    # weights: np.ndarray,
) -> dict:
    """Creates descriptive statistics of a Network.

    Calculates the following statistics for the network:
        avg_connectedness: Average connectedness of the network table.
        asymmetry: Asymmetry of network table.
        asymmetry_offdiag: Off-diagonal asymmetry of network table.
        concentration_out_connectedness:
        concentration_out_eigenvector_centrality:
        concentration_out_page_rank:
        concentration_out_connectedness_herfindahl:
        concentration_out_eigenvector_centrality_herfindahl:
        concentration_out_page_rank_herfindahl:
        amplification:

    Args:
        network: Network object to be described.
     #   weights: A vector indicating the weights of each node in the aggregate.

    Returns:
        stats: Key, value pairs of the calculated statistics.
    """
    stats = {}
    stats[f"avg_connectedness"] = network.average_connectedness()
    stats[f"asymmetry"] = matrix_asymmetry(network.adjacency_matrix)
    stats[f"asymmetry_offdiag"] = matrix_asymmetry(
        network.adjacency_matrix, drop_diag=True
    )
    stats[f"concentration_out_connectedness_powerlaw"] = power_law_exponent(
        network.out_connectedness(),
        invert=True,
    )
    stats[f"concentration_full_out_connectedness_powerlaw"] = power_law_exponent(
        network.out_connectedness(others_only=False),
        invert=True,
    )
    stats[f"concentration_out_eigenvector_centrality_powerlaw"] = power_law_exponent(
        network.out_eigenvector_centrality(),
        invert=True,
    )
    stats[f"concentration_out_page_rank_powerlaw"] = power_law_exponent(
        network.out_page_rank(),
        invert=True,
    )
    stats[f"concentration_out_connectedness_herfindahl"] = herfindahl_index(
        network.out_connectedness(),
    )
    stats[f"concentration_full_out_connectedness_herfindahl"] = herfindahl_index(
        network.out_connectedness(others_only=False),
    )
    stats[f"concentration_out_eigenvector_centrality_herfindahl"] = herfindahl_index(
        network.out_eigenvector_centrality(),
    )
    stats[f"concentration_out_page_rank_herfindahl"] = herfindahl_index(
        network.out_page_rank(),
    )
    # stats[f"amplification"] = (
    #     network.amplification_factor().squeeze() * (weights / weights.sum()).squeeze()
    # ).sum()
    return stats


def describe_fevd(
    fevd: euraculus.fevd.FEVD,
    horizon: int,
    data: pd.DataFrame,
    weights: np.ndarray,
) -> dict:
    """Creates descriptive statistics of a FEVD.

    Calculates the following statistics for the FEVD, then describes some FEVD network tables:
        innovation_diagonality_test_stat': Ledoit-Wolf test statistic for diagonality of innovations.
        innovation_diagonality_p_value': P-value for Ledoit-Wolf test statistic.

    Args:
        fevd: Forecast Error Variance Decomposition to be described.
        horizon: Horizon to calculate the descriptive statistics with.
        data: Data the estimation is performed on.
        weights: A vector indicating the weights of each node in the aggregate.

    Returns:
        stats: Key, value pairs of the calculated statistics.

    """
    # direct analysis of fevd
    stats = {}
    (
        stats["innovation_diagonality_test_stat"],
        stats["innovation_diagonality_p_value"],
    ) = fevd.test_diagonal_generalized_innovations(t_observations=data.shape[0])

    # analyse networks derived from fevd
    tables = [
        ("fevd", None),
        ("fev", None),
        ("fevd", weights),
        ("fev", weights),
    ]
    for (table_name, weights) in tables:
        network = fevd.to_network(
            horizon=horizon,
            table_name=table_name,
            normalize=False,
            weights=weights,
        )
        network_stats = describe_network(network)
        weighted = "" if weights is None else "w"
        network_stats = {
            f"{weighted}{table_name}_{k}": v for k, v in network_stats.items()
        }
        stats.update(network_stats)
    return stats


def collect_data_estimates(
    df_historic: pd.DataFrame,
    df_future: pd.DataFrame,
    df_rf: pd.DataFrame,
    analysis_windows: list,
) -> pd.DataFrame:
    """Calculate return and variance estimates for various horizons.

    Args:
        df_historic: Dataframe with past returns.
        df_future: Dataframe with subsequent returns.
        df_rf: Dataframe with risk-free returns.
        analysis_windows: List of forecast horizons in months.

    Returns:
        estimates: Extracted estimates in a DataFrame.
    """
    estimates = pd.DataFrame(index=df_historic.columns)

    # historic
    df_historic -= df_rf.loc[df_historic.index].values
    estimates["ret_excess"] = (1 + df_historic).prod() - 1
    estimates["var_annual"] = df_historic.var() * 252

    # forecasts
    if df_future is not None:
        df_future -= df_rf.loc[df_future.index].values
        for window_length in analysis_windows:
            if (
                months_difference(
                    end_date=df_future.index.max(),
                    start_date=df_historic.index.max(),
                )
                >= window_length
            ):
                end_date = df_historic.index.max() + relativedelta(
                    months=window_length, day=31
                )
                df_window = df_future[df_future.index <= end_date]
                estimates[f"ret_excess_next{window_length}M"] = (
                    1 + df_window
                ).prod() - 1
                estimates[f"var_annual_next{window_length}M"] = df_window.var() * 252

    return estimates


def collect_var_estimates(
    var: euraculus.var.VAR,
    var_data: pd.DataFrame,
    factor_data: pd.DataFrame,
) -> pd.DataFrame:
    """Extract estimates from a VAR.

    Extracts the following estimates on asset level:
        var_intercept: The intercept in the VAR model.
        var_factor_loadings: The factor loadings in the factor VAR.
        var_mean_abs_in: Average value of in connections in VAR.
        var_mean_abs_out: Average absolute value of out connections in VAR.
        var_factor_residual_variance: Variance of factor residuals.
        var_residual_variance:Variance of VAR model residuals.

    Args:
        var: Vector autoregression to extract estimates from.
        var_data: Data the estimation is performed on.
        factor_data: Factor data for the estiamtion period.

    Returns:
        estimates: Extracted estimates in a DataFrame.

    """
    estimates = pd.DataFrame(index=var_data.columns)
    estimates["var_intercept"] = var.intercepts_
    for i_factor, factor in enumerate(factor_data.columns):
        estimates[f"var_factor_loadings_{factor}"] = var.factor_loadings_[:, i_factor]
    estimates["var_mean_abs_in"] = (
        abs(var.var_1_matrix_).sum(axis=1) - abs(np.diag(var.var_1_matrix_))
    ) / (var.var_1_matrix_.shape[0] - 1)
    estimates["var_mean_abs_out"] = (
        abs(var.var_1_matrix_).sum(axis=0) - abs(np.diag(var.var_1_matrix_))
    ) / (var.var_1_matrix_.shape[0] - 1)
    factor_residuals = var.factor_residuals(var_data=var_data, factor_data=factor_data)
    estimates["var_factor_residual_variance"] = np.diag(factor_residuals.cov())
    residuals = var.residuals(var_data=var_data, factor_data=factor_data)
    estimates["var_residual_variance"] = np.diag(residuals.cov())
    estimates["var_systematic_variance"] = var.systematic_variances(
        factor_data=factor_data
    )
    return estimates


def collect_cov_estimates(
    cov: euraculus.covar.GLASSO,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Extract estimates from a GLASSO Covariance estimation.

    Extracts the following estimates on asset level:
        cov_variance: Variance estimates.
        cov_mean_corr: Average estimated correlation with other series.

    Args:
        cov: GLASSO covariance object to extract estimates from.
        data: Data the estimation is performed on.

    Returns:
        estimates: Extracted estimates in a DataFrame.

    """
    estimates = pd.DataFrame(index=data.columns)
    estimates["cov_variance"] = np.diag(cov.covariance_)
    estimates["cov_mean_corr"] = (
        euraculus.utils.cov_to_corr(cov.covariance_).sum(axis=1) - 1
    ) / (cov.covariance_.shape[0] - 1)
    return estimates


def collect_network_estimates(
    network: euraculus.network.Network,
    data: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Extract estimates from a Network.

    Extracts the following estimates on node level for each table and weighting:
        in_connectedness: Sum of incoming links.
        out_connectedness: Sum of outgoing links.
        self_connectedness: Link with itself.
        total_connectedness: Total links of each node (incoming and outgoing).
        in_concentration: Concentration of incoming links.
        out_concentration: Concentration of outgoing links.
        in_eigenvector_centrality: Eigenvector centrality of incoming links.
        out_eigenvector_centrality: Eigenvector centrality of outgoing links.
        in_page_rank_equal: Page rank of incoming links without personalisation.
        out_page_rank_equal: Page rank of outgoing links without personalisation.
        in_page_rank_85: Page rank of incoming links with alpha 0.85.
        out_page_rank_85: Page rank of outgoing links with alpha 0.85.
        in_page_rank_95: Page rank of incoming links with alpha 0.95.
        out_page_rank_95: Page rank of outgoing links with alpha 0.95.
        amplification_factor:
        absorption_rate:

    Args:
        fevd: Forecast Error Variance Decomposition to be described.
        data: Data the estimation is performed on.
        weights: A vector indicating the weights of each node in the aggregate.

    Returns:
        estimates: Extracted estimates in a DataFrame.
    """
    estimates = pd.DataFrame(index=data.columns)
    estimates[f"in_connectedness"] = network.in_connectedness()
    estimates[f"out_connectedness"] = network.out_connectedness()
    estimates[f"full_in_connectedness"] = network.in_connectedness(others_only=False)
    estimates[f"full_out_connectedness"] = network.out_connectedness(others_only=False)
    estimates[f"self_connectedness"] = network.self_connectedness()
    estimates[f"total_connectedness"] = network.total_connectedness()
    estimates[f"in_concentration"] = network.in_concentration()
    estimates[f"out_concentration"] = network.out_concentration()
    estimates[f"full_in_concentration"] = network.in_concentration(others_only=False)
    estimates[f"full_out_concentration"] = network.out_concentration(others_only=False)
    estimates[f"in_eigenvector_centrality"] = network.in_eigenvector_centrality()
    estimates[f"out_eigenvector_centrality"] = network.out_eigenvector_centrality()
    estimates[f"in_page_rank_equal"] = network.in_page_rank(weights=None, alpha=0.85)
    estimates[f"out_page_rank_equal"] = network.out_page_rank(weights=None, alpha=0.85)
    estimates[f"in_page_rank_85"] = network.in_page_rank(weights=weights, alpha=0.85)
    estimates[f"out_page_rank_85"] = network.out_page_rank(weights=weights, alpha=0.85)
    estimates[f"in_page_rank_95"] = network.in_page_rank(weights=weights, alpha=0.95)
    estimates[f"out_page_rank_95"] = network.out_page_rank(weights=weights, alpha=0.95)
    estimates[f"amplification_factor"] = network.amplification_factor()
    estimates[f"absorption_rate"] = network.absorption_rate()
    return estimates


def collect_fevd_estimates(
    fevd: euraculus.fevd.FEVD,
    horizon: int,
    data: pd.DataFrame,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Extract estimates from a Forecast Error Variance Decomposition network.

    Extracts estimates on node level from a set of FEVD tables.

    Args:
        fevd: Forecast Error Variance Decomposition to be described.
        horizon: Horizon to calculate some estimates with.
        data: Data the estimation is performed on.
        weights: A vector indicating the weights of each node in the aggregate.

    Returns:
        estimates: Extracted estimates in a DataFrame.
    """
    estimates = pd.DataFrame(index=data.columns)
    tables = [
        ("fevd", None),
        ("fev", None),
        ("fevd", weights),
        ("fev", weights),
    ]
    for (table_name, weights) in tables:
        network = fevd.to_network(
            horizon=horizon,
            table_name=table_name,
            normalize=False,
            weights=weights,
        )
        network_estimates = collect_network_estimates(
            network=network, weights=weights, data=data
        )
        weighted = "" if weights is None else "w"
        network_estimates = network_estimates.add_prefix(f"{weighted}{table_name}_")
        estimates = estimates.join(network_estimates)
    return estimates
