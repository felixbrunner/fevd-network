import warnings
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
from euraculus.utils import matrix_asymmetry, shrinkage_factor
import datetime as dt

import euraculus


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
        columns=["ticker", "comnam", "last_size", "mean_size"],
    )

    # prepare asset data
    df_vola = np.sqrt(df_var)
    df_noisevola = np.sqrt(df_noisevar)
    df_lagged_mcap = df_mcap / (df_ret + 1)
    df_log_vola = prepare_log_data(df_data=df_vola, df_fill=df_noisevola)
    df_log_mcap = log_replace(df=df_lagged_mcap, method="ffill")
    df_log_mcap_vola = df_log_vola + df_log_mcap

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

    df_spy = data.load_spy_data().reindex(df_var.index)
    spy_factor = prepare_spy_factor(df_spy)
    df_factors = df_factors.join(spy_factor)
    for ticker in ["^VIX", "DX-Y.NYB", "^TNX"]:
        df_yahoo = data.load_yahoo(ticker).reindex(df_var.index)
        factor = prepare_yahoo_factor(df_yahoo).rename(ticker)
        df_factors = df_factors.join(factor)

    return (df_info, df_log_mcap_vola, df_factors)


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


def estimate_fevd(
    var_data: pd.DataFrame,
    factor_data: pd.DataFrame,
    var_grid: dict,
    cov_grid: dict,
) -> tuple:
    """Perform all estimation steps necessary to construct FEVD.

    Args:
        var_data:
        factor_data:
        var_grid:
        cov_grid:

    Returns:
        var_cv:
        var:
        cov_cv:
        cov:
        fevd:

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


def describe_fevd(
    fevd: euraculus.fevd.FEVD,
    horizon: int,
    data: pd.DataFrame,
) -> dict:
    """Creates descriptive statistics of a GLASSO Covariance estimation.

    Calculates the following statistics:
        fevd_avg_connectedness: Average connectedness of FEVD table.
        fevd_avg_connectedness_normalized: Average connectedness of row-normalized FEVD table.
        fev_avg_connectedness: Average connectedness of FEV table.
        irv_avg_connectedness: Average connectedness of IRV table.
        fevd_asymmetry: Asymmetry of FEVD table.
        fevd_asymmetry_normalized: Asymmetry of row-normalized FEVD table.
        fev_asymmetry: Asymmetry of FEV table.
        irv_asymmetry: Asymmetry of IRV table.
        innovation_diagonality_test_stat': Ledoit-Wolf test statistic for diagonality of innovations.
        innovation_diagonality_p_value': P-value for Ledoit-Wolf test statistic.

    Args:
        fevd: Forecast Error Variance Decomposition to be described.
        horizon: Horizon to calculate the descriptive statistics with.
        data: Data the estimation is performed on.

    Returns:
        stats: Key, value pairs of the calculated statistics.

    """
    stats = {
        "fevd_avg_connectedness": fevd.average_connectedness(
            horizon=horizon, table_name="fevd", normalize=False
        ),
        "fevd_avg_connectedness_normalized": fevd.average_connectedness(
            horizon=horizon, table_name="fevd", normalize=True
        ),
        "fev_avg_connectedness": fevd.average_connectedness(
            horizon=horizon, table_name="fev", normalize=False
        ),
        "irv_avg_connectedness": fevd.average_connectedness(
            horizon=horizon, table_name="irv", normalize=False
        ),
        "fevd_asymmetry": matrix_asymmetry(
            fevd.forecast_error_variance_decomposition(horizon=horizon, normalize=False)
        ),
        "fevd_asymmetry_normalized": matrix_asymmetry(
            fevd.forecast_error_variance_decomposition(horizon=horizon, normalize=True)
        ),
        "fev_asymmetry": matrix_asymmetry(
            fevd.forecast_error_variances(horizon=horizon)
        ),
        "irv_asymmetry": matrix_asymmetry(
            fevd.innovation_response_variances(horizon=horizon)
        ),
    }
    (
        stats["innovation_diagonality_test_stat"],
        stats["innovation_diagonality_p_value"],
    ) = fevd.test_diagonal_generalized_innovations(t_observations=data.shape[0])
    return stats


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


def collect_fevd_estimates(
    fevd: euraculus.fevd.FEVD,
    horizon: int,
    data: pd.DataFrame,
    # sizes: pd.Series,
) -> pd.DataFrame:
    """Extract estimates from a Forecast Error Variance Decomposition network.

    Extracts the following estimates on asset level:
        fev_total:
        fev_others:
        fev_self:
        fevd_in_connectedness:
        fevd_out_connectedness:
        fevd_eigenvector_centrality:
        fevd_closeness_centrality:
        fevd_in_entropy:
        fevd_in_connectedness_normalized:
        fevd_out_connectedness_normalized:
        fevd_eigenvector_centrality_normalized:
        fevd_closeness_centrality_normalized:
        fu_total:
        fu_others:
        fu_self:
        fud_in_connectedness:
        fud_out_connectedness:
        fud_eigenvector_centrality:
        fud_closeness_centrality:
        fud_in_entropy:
        fud_in_connectedness_normalized:
        fud_out_connectedness_normalized:
        fud_eigenvector_centrality_normalized:
        fud_closeness_centrality_normalized:
        # irv_index_decomposition_absolute:
        # irv_index_decomposition_shares:

    Args:
        fevd: Forecast Error Variance Decomposition to be described.
        horizon: Horizon to calculate some estimates with.
        # data: Data the estimation is performed on.
        # sizes: Firm sizes to define weights.

    Returns:
        estimates: Extracted estimates in a DataFrame.

    """

    estimates = pd.DataFrame(index=data.columns)

    tables = [("fevd", False), ("fevd", True), ("fev", False), ("irv", False)]

    for (table, normalize) in tables:
        estimates[f"{table}_in_connectedness"] = fevd.in_connectedness(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[f"{table}_out_connectedness"] = fevd.out_connectedness(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[f"{table}_self_connectedness"] = fevd.self_connectedness(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[f"{table}_total_connectedness"] = fevd.total_connectedness(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[f"{table}_in_entropy"] = fevd.in_entropy(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[f"{table}_out_entropy"] = fevd.out_entropy(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[
            f"{table}_in_eigenvector_centrality"
        ] = fevd.in_eigenvector_centrality(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[
            f"{table}_out_eigenvector_centrality"
        ] = fevd.out_eigenvector_centrality(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[f"{table}_in_page_rank"] = fevd.in_page_rank(
            horizon=horizon, table_name=table, normalize=normalize
        )
        estimates[f"{table}_out_page_rank"] = fevd.out_page_rank(
            horizon=horizon, table_name=table, normalize=normalize
        )

    return estimates
