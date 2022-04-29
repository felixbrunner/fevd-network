import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import GridSearchCV

import euraculus

# import matplotlib.pyplot as plt


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
        lambda:
        kappa:
        ini_lambda:
        ini_kappa:
        var_matrix_density:
        var_mean_connection:
        var_mean_abs_connection:
        var_asymmetry:
        var_r2:
        var_r2_ols:
        var_df_used:
        var_nonzero_shrinkage:
        var_full_shrinkage:
        var_cv_loss:
        var_train_loss:

    Args:
        var: Vector autoregression to be described.
        var_cv: Cross-validation from the preceeding estimation.
        data: Data the estimation is performed on.

    Returns:
        stats: Key, value pairs of the calculated statistics.

    """
    ols_var = var.copy()
    if factor_data is not None:
        ols_var.fit_ols(var_data=var_data, factor_data=factor_data)
        stats = {
            "lambda": var_cv.best_params_["lambdau"],
            "kappa": var_cv.best_params_["alpha"],
            "ini_lambda": var_cv.best_estimator_.ini_lambdau,
            "ini_kappa": var_cv.best_estimator_.ini_lambdau,
            "var_matrix_density": (var.var_1_matrix_ != 0).sum()
            / var.var_1_matrix_.size,
            "var_mean_connection": var.var_1_matrix_.mean(),
            "var_mean_abs_connection": abs(var.var_1_matrix_).mean(),
            "var_asymmetry": euraculus.utils.matrix_asymmetry(M=var.var_1_matrix_),
            "var_r2": var.r2(var_data=var_data, factor_data=factor_data),
            "var_r2_ols": ols_var.r2(var_data=var_data, factor_data=factor_data),
            "var_factor_r2": var.factor_r2(var_data=var_data, factor_data=factor_data),
            "var_factor_r2_ols": ols_var.factor_r2(
                var_data=var_data, factor_data=factor_data
            ),
            "var_df_used": var.df_used_,
            "var_nonzero_shrinkage": euraculus.utils.shrinkage_factor(
                array=var.var_1_matrix_,
                benchmark_array=ols_var.var_1_matrix_,
                drop_zeros=True,
            ),
            "var_full_shrinkage": euraculus.utils.shrinkage_factor(
                array=var.var_1_matrix_,
                benchmark_array=ols_var.var_1_matrix_,
                drop_zeros=False,
            ),
            "var_factor_shrinkage": euraculus.utils.shrinkage_factor(
                array=var.factor_loadings_,
                benchmark_array=ols_var.factor_loadings_,
                drop_zeros=True,
            ),
            "var_full_factor_shrinkage": euraculus.utils.shrinkage_factor(
                array=var.factor_loadings_,
                benchmark_array=ols_var.factor_loadings_,
                drop_zeros=False,
            ),
            "var_cv_loss": -var_cv.best_score_,
            "var_train_loss": -var_cv.cv_results_["mean_train_score"][
                var_cv.best_index_
            ],
        }
    else:
        ols_var.fit_ols(var_data=var_data)
        stats = {
            "lambda": var_cv.best_params_["lambdau"],
            "kappa": var_cv.best_params_["alpha"],
            "ini_lambda": var_cv.best_estimator_.ini_lambdau,
            "ini_kappa": var_cv.best_estimator_.ini_lambdau,
            "var_matrix_density": (var.var_1_matrix_ != 0).sum()
            / var.var_1_matrix_.size,
            "var_mean_connection": var.var_1_matrix_.mean(),
            "var_mean_abs_connection": abs(var.var_1_matrix_).mean(),
            "var_asymmetry": euraculus.utils.matrix_asymmetry(M=var.var_1_matrix_),
            "var_r2": var.r2(var_data=var_data),
            "var_r2_ols": ols_var.r2(var_data=var_data),
            "var_df_used": var.df_used_,
            "var_nonzero_shrinkage": euraculus.utils.shrinkage_factor(
                array=var.var_1_matrix_,
                benchmark_array=ols_var.var_1_matrix_,
                drop_zeros=True,
            ),
            "var_full_shrinkage": euraculus.utils.shrinkage_factor(
                array=var.var_1_matrix_,
                benchmark_array=ols_var.var_1_matrix_,
                drop_zeros=False,
            ),
            "var_cv_loss": -var_cv.best_score_,
            "var_train_loss": -var_cv.cv_results_["mean_train_score"][
                var_cv.best_index_
            ],
        }

    return stats


def describe_cov(
    cov: euraculus.covar.GLASSO,
    cov_cv: GridSearchCV,
    data: pd.DataFrame,
) -> dict:
    """Creates descriptive statistics of a GLASSO Covariance estimation.

    Calculates the following statistics:
        rho:
        cov_mean_likelihood:
        cov_mean_likelihood_sample_estimate:
        covar_density:
        precision_density:
        covar_nonzero_shrinkage:
        covar_full_shrinkage:
        precision_nonzero_shrinkage:
        precision_full_shrinkage:
        covar_cv_loss:
        covar_train_loss:

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
        "covar_nonzero_shrinkage": euraculus.utils.shrinkage_factor(
            array=cov.covariance_,
            benchmark_array=data.cov().values,
            drop_zeros=True,
        ),
        "covar_full_shrinkage": euraculus.utils.shrinkage_factor(
            array=cov.covariance_,
            benchmark_array=data.cov().values,
            drop_zeros=False,
        ),
        "precision_nonzero_shrinkage": euraculus.utils.shrinkage_factor(
            array=cov.precision_,
            benchmark_array=np.linalg.inv(data.cov().values),
            drop_zeros=True,
        ),
        "precision_full_shrinkage": euraculus.utils.shrinkage_factor(
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
        fev_avg_connectedness:
        fev_avg_connectedness_normalised:
        fu_avg_connectedness:
        fu_avg_connectedness_normalised:
        fev_asymmetry:
        fev_asymmetry_normalised:
        fu_asymmetry:
        fu_asymmetry_normalised:
        innovation_diagonality_test_stat:
        innovation_diagonality_p_value:

    Args:
        fevd: Forecast Error Variance Decomposition to be described.
        horizon: Horizon to calculate the descriptive statistics with.
        data: Data the estimation is performed on.

    Returns:
        stats: Key, value pairs of the calculated statistics.

    """
    stats = {
        "fev_avg_connectedness": fevd.average_connectedness(
            horizon, normalise=False, network="fev"
        ),
        "fev_avg_connectedness_normalised": fevd.average_connectedness(
            horizon, normalise=True, network="fev"
        ),
        "fu_avg_connectedness": fevd.average_connectedness(
            horizon, normalise=False, network="fu"
        ),
        "fu_avg_connectedness_normalised": fevd.average_connectedness(
            horizon, normalise=True, network="fu"
        ),
        "fev_asymmetry": euraculus.utils.matrix_asymmetry(
            M=fevd.decompose_fev(horizon, normalise=False)
        ),
        "fev_asymmetry_normalised": euraculus.utils.matrix_asymmetry(
            M=fevd.decompose_fev(horizon, normalise=True)
        ),
        "fu_asymmetry": euraculus.utils.matrix_asymmetry(
            M=fevd.decompose_fu(horizon, normalise=False)
        ),
        "fu_asymmetry_normalised": euraculus.utils.matrix_asymmetry(
            M=fevd.decompose_fu(horizon, normalise=True)
        ),
    }
    (
        stats["innovation_diagonality_test_stat"],
        stats["innovation_diagonality_p_value"],
    ) = fevd.test_diagonal_generalized_innovations(t_observations=data.shape[0])
    return stats


def collect_var_estimates(
    var: euraculus.var.VAR,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Extract estimates from a VAR.

    Extracts the following estimates on asset level:
        var_intercept:
        mean_abs_var_in:
        mean_abs_var_out:

    Args:
        var: Vector autoregression to extract estimates from.
        data: Data the estimation is performed on.

    Returns:
        estimates: Extracted estimates in a DataFrame.

    """
    estimates = pd.DataFrame(index=data.columns)
    estimates["var_intercept"] = var.intercepts_
    if hasattr(var, "factor_loadings_"):
        estimates["var_factor_loadings_"] = var.factor_loadings_
    estimates["mean_abs_var_in"] = (
        abs(var.var_1_matrix_).sum(axis=1) - abs(np.diag(var.var_1_matrix_))
    ) / (var.var_1_matrix_.shape[0] - 1)
    estimates["mean_abs_var_out"] = (
        abs(var.var_1_matrix_).sum(axis=0) - abs(np.diag(var.var_1_matrix_))
    ) / (var.var_1_matrix_.shape[0] - 1)
    return estimates


def collect_cov_estimates(
    cov: euraculus.covar.GLASSO,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Extract estimates from a GLASSO Covariance estimation.

    Extracts the following estimates on asset level:
        residual_variance:
        mean_resid_corr:

    Args:
        cov: GLASSO covariance object to extract estimates from.
        data: Data the estimation is performed on.

    Returns:
        estimates: Extracted estimates in a DataFrame.

    """
    estimates = pd.DataFrame(index=data.columns)
    estimates["residual_variance"] = np.diag(cov.covariance_)
    estimates["mean_resid_corr"] = (
        euraculus.utils.cov_to_corr(cov.covariance_).sum(axis=1) - 1
    ) / (cov.covariance_.shape[0] - 1)
    return estimates


def collect_fevd_estimates(
    fevd: euraculus.fevd.FEVD,
    horizon: int,
    data: pd.DataFrame,
    sizes: pd.Series,
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
        fevd_in_connectedness_normalised:
        fevd_out_connectedness_normalised:
        fevd_eigenvector_centrality_normalised:
        fevd_closeness_centrality_normalised:
        fu_total:
        fu_others:
        fu_self:
        fud_in_connectedness:
        fud_out_connectedness:
        fud_eigenvector_centrality:
        fud_closeness_centrality:
        fud_in_entropy:
        fud_in_connectedness_normalised:
        fud_out_connectedness_normalised:
        fud_eigenvector_centrality_normalised:
        fud_closeness_centrality_normalised:
        irv_index_decomposition_absolute:
        irv_index_decomposition_shares:

    Args:
        fevd: Forecast Error Variance Decomposition to be described.
        horizon: Horizon to calculate some estimates with.
        data: Data the estimation is performed on.
        sizes: Firm sizes to define weights.

    Returns:
        estimates: Extracted estimates in a DataFrame.

    """

    estimates = pd.DataFrame(index=data.columns)

    # fev
    estimates["fev_total"] = fevd.fev_total(horizon=horizon)
    estimates["fev_others"] = fevd.fev_others(horizon=horizon)
    estimates["fev_self"] = fevd.fev_self(horizon=horizon)

    # non-normalised fevd
    estimates["fevd_in_connectedness"] = fevd.in_connectedness(
        horizon=horizon, normalise=False, network="fev"
    )
    estimates["fevd_out_connectedness"] = fevd.out_connectedness(
        horizon=horizon, normalise=False, network="fev"
    )
    estimates["fevd_eigenvector_centrality"] = list(
        nx.eigenvector_centrality(
            fevd.to_fev_graph(horizon, normalise=False), weight="weight", max_iter=1000
        ).values()
    )
    estimates["fevd_closeness_centrality"] = list(
        nx.closeness_centrality(
            fevd.to_fev_graph(horizon, normalise=False), distance="weight"
        ).values()
    )
    estimates["fevd_in_entropy"] = fevd.in_entropy(
        horizon=horizon, normalise=True, network="fev"
    )

    # normalised fevd
    estimates["fevd_in_connectedness_normalised"] = fevd.in_connectedness(
        horizon=horizon, normalise=True, network="fev"
    )
    estimates["fevd_out_connectedness_normalised"] = fevd.out_connectedness(
        horizon=horizon, normalise=True, network="fev"
    )
    estimates["fevd_eigenvector_centrality_normalised"] = list(
        nx.eigenvector_centrality(
            fevd.to_fev_graph(horizon, normalise=True), weight="weight", max_iter=1000
        ).values()
    )
    estimates["fevd_closeness_centrality_normalised"] = list(
        nx.closeness_centrality(
            fevd.to_fev_graph(horizon, normalise=True), distance="weight"
        ).values()
    )

    # fu
    estimates["fu_total"] = fevd.fu_total(horizon=horizon)
    estimates["fu_others"] = fevd.fu_others(horizon=horizon)
    estimates["fu_self"] = fevd.fu_self(horizon=horizon)

    # non-normalised fud
    estimates["fud_in_connectedness"] = fevd.in_connectedness(
        horizon=horizon, normalise=False, network="fu"
    )
    estimates["fud_out_connectedness"] = fevd.out_connectedness(
        horizon=horizon, normalise=False, network="fu"
    )
    estimates["fud_eigenvector_centrality"] = list(
        nx.eigenvector_centrality(
            fevd.to_fu_graph(horizon, normalise=False), weight="weight", max_iter=1000
        ).values()
    )
    estimates["fud_closeness_centrality"] = list(
        nx.closeness_centrality(
            fevd.to_fu_graph(horizon, normalise=False), distance="weight"
        ).values()
    )
    estimates["fud_in_entropy"] = fevd.in_entropy(
        horizon=horizon, normalise=True, network="fu"
    )

    # normalised fud
    estimates["fud_in_connectedness_normalised"] = fevd.in_connectedness(
        horizon=horizon, normalise=True, network="fu"
    )
    estimates["fud_out_connectedness_normalised"] = fevd.out_connectedness(
        horizon=horizon, normalise=True, network="fu"
    )
    estimates["fud_eigenvector_centrality_normalised"] = list(
        nx.eigenvector_centrality(
            fevd.to_fu_graph(horizon, normalise=True), weight="weight", max_iter=1000
        ).values()
    )
    estimates["fud_closeness_centrality_normalised"] = list(
        nx.closeness_centrality(
            fevd.to_fu_graph(horizon, normalise=True), distance="weight"
        ).values()
    )

    # innovation response variance
    estimates["irv_index_decomposition_absolute"] = fevd.index_variance_decomposition(
        horizon=horizon, weights=sizes
    )
    estimates["irv_index_decomposition_shares"] = fevd.index_variance_decomposition(
        horizon=horizon, weights=sizes / sizes.sum()
    )

    return estimates


##############################################################################


# def run_estimation_short(year, month, var_grid, cov_grid, horizon):
#     # data
#     df_log_idio_var, df_var, df_spy_var = load_preprocess_short(year, month)

#     # estimate & combine
#     var_cv, var = estimate_var_short(df_log_idio_var, var_grid)
#     residuals = var.residuals(df_log_idio_var)
#     cov_cv, cov = estimate_cov_short(residuals, cov_grid)
#     fevd = make_fevd(var, cov, horizon)

#     # estimation description
#     data_desc = describe_data(df_log_idio_var)
#     var_desc = describe_var_short(var_cv, var, df_log_idio_var)
#     cov_desc = describe_cov_short(cov_cv, cov, residuals)
#     fevd_desc = describe_fevd(fevd, horizon)
#     desc = data_desc.append(var_desc).append(cov_desc).append(fevd_desc)

#     # collect estimate data
#     var_data = collect_var_data_short(var, df_log_idio_var)
#     cov_data = collect_cov_data_short(cov, var, df_log_idio_var)
#     fevd_data = collect_fevd_data_short(fevd, horizon, var, df_log_idio_var)
#     network_data = var_data.join(cov_data).join(fevd_data)

#     # save estimates
#     pd.DataFrame(
#         data=var.var_1_matrix_,
#         index=df_log_idio_var.columns,
#         columns=df_log_idio_var.columns,
#     ).to_csv("../data/estimated/monthly/{}/{}/var_matrix.csv".format(year, month))
#     pd.DataFrame(
#         data=cov.covariance_,
#         index=df_log_idio_var.columns,
#         columns=df_log_idio_var.columns,
#     ).to_csv("../data/estimated/monthly/{}/{}/cov_matrix.csv".format(year, month))
#     pd.DataFrame(
#         data=fevd.decompose_fev(horizon=horizon, normalise=True),
#         index=df_log_idio_var.columns,
#         columns=df_log_idio_var.columns,
#     ).to_csv(
#         "../data/estimated/monthly/{}/{}/fevd_matrix_normalised.csv".format(year, month)
#     )

#     # save estimation data
#     desc.to_csv("../data/estimated/monthly/{}/{}/desc.csv".format(year, month))
#     network_data.to_csv(
#         "../data/estimated/monthly/{}/{}/network_data.csv".format(year, month)
#     )

#     return (df_log_idio_var, var_cv, var, cov_cv, cov, fevd, desc, network_data)


# def load_preprocess_short(year, month):
#     # load
#     df_data = euraculus.loader.load_monthly_estimation_data(year, month, column=None)
#     df_spy = euraculus.loader.load_spy()

#     # select
#     df_idio_var = df_data["idiosyncratic"].unstack()
#     df_var = df_data["total"].unstack()
#     df_spy_var = df_spy[df_spy.index.isin(df_idio_var.index)][["var"]]

#     # transform
#     df_log_idio_var = euraculus.utils.log_replace(df_idio_var, method="min")
#     return (df_log_idio_var, df_var, df_spy_var)


# def estimate_var_short(data, var_grid):
#     var = euraculus.VAR(add_intercepts=True, p_lags=1)
#     var_cv = var.fit_adaptive_elastic_net_cv(data, grid=var_grid, return_cv=True)
#     return (var_cv, var)


# def estimate_cov_short_ate(data, cov_grid):
#     cov_cv = GridSearchCV(
#         euraculus.covar.AdaptiveThresholdEstimator(),
#         param_grid=cov_grid,
#         cv=12,
#         n_jobs=-1,
#         verbose=1,
#         return_train_score=True,
#     ).fit(data)
#     cov = cov_cv.best_estimator_
#     return (cov_cv, cov)


# def estimate_cov_short(data, cov_grid):
#     cov_cv = GridSearchCV(
#         euraculus.covar.GLASSO(max_iter=200),
#         param_grid=cov_grid,
#         cv=12,
#         n_jobs=-1,
#         verbose=1,
#         return_train_score=True,
#     ).fit(data)
#     cov = cov_cv.best_estimator_
#     return (cov_cv, cov)


# def make_fevd(var, cov, horizon):
#     fevd = euraculus.FEVD(var.var_1_matrix_, cov.covariance_)
#     return fevd


######## CODE STILL IN USE ######################################################################


# def run_estimation(year, var_grid, cov_grid, horizon):
#     # data
#     preprocessed_data, df_volas, df_spy_vola = load_preprocess(year)
#     tickers = euraculus.loader.load_year_tickers(year)
#     column_to_ticker = dict(
#         pd.read_csv("../data/processed/annual/{}/tickers.csv".format(year))[
#             "permno_to_ticker"
#         ]
#     )

#     # estimate & combine
#     var_cv, var = estimate_var(
#         year, preprocessed_data, var_grid, weighting_method="ridge"
#     )
#     cov_cv, cov = estimate_cov(var.residuals_, cov_grid)
#     fevd = make_fevd(var, cov, horizon)

#     # estimation description
#     data_desc = describe_data(preprocessed_data)
#     var_desc = describe_var(var_cv, var)
#     cov_desc = describe_cov(cov_cv, cov, var)
#     fevd_desc = describe_fevd(fevd, horizon)
#     desc = data_desc.append(var_desc).append(cov_desc).append(fevd_desc)
#     desc.index.name = "year"

#     # collect estimate data
#     var_data = collect_var_data(var)
#     cov_data = collect_cov_data(cov, var)
#     fevd_data = collect_fevd_data(fevd, horizon, var)
#     network_data = var_data.join(cov_data).join(fevd_data)

#     # save estimation data
#     desc.to_csv("../data/estimated/annual/{}/desc.csv".format(year))
#     network_data.to_csv("../data/estimated/annual/{}/network_data.csv".format(year))

#     # plots
#     create_data_plots(
#         year, preprocessed_data, df_volas, df_spy_vola
#     )  # data description
#     create_var_plots(year, var_cv, var)  # VAR
#     create_cov_plots(year, cov_cv, cov)  # Covariance matrix
#     create_fevd_plots(year, fevd, horizon, column_to_ticker)  # FEVD

#     return (preprocessed_data, var_cv, var, cov_cv, cov, fevd, desc, network_data)


# def load_preprocess(year):
#     data = euraculus.loader.load_year(year, data="idio_var")
#     df_volas = euraculus.loader.load_year(year, data="volas")
#     df_spy_vola = euraculus.loader.load_spy_vola(year)
#     euraculus.plot.missing_data(
#         data, save_path="../reports/figures/annual/{}/matrix_missing.pdf".format(year)
#     )
#     euraculus.plot.histogram(
#         data.fillna(0).stack(),
#         title="Distribution of Raw Data",
#         drop_tails=0.01,
#         save_path="../reports/figures/annual/{}/histogram_raw_data.pdf".format(year),
#     )
#     euraculus.plot.var_timeseries(
#         data,
#         total_var=df_volas ** 2,
#         index_var=df_spy_vola ** 2,
#         save_path="../reports/figures/annual/{}/variance_decomposition.pdf".format(
#             year
#         ),
#     )
#     data = euraculus.utils.log_replace(data, method="min")
#     return (data, df_volas, df_spy_vola)


# def estimate_var(year, data, var_grid, weighting_method="OLS"):
#     var = euraculus.VAR(data, p_lags=1)
#     var.fit_OLS()
#     euraculus.plot.corr_heatmap(
#         var.var_1_matrix_,
#         title="Non-regularized VAR(1) coefficient matrix (OLS)",
#         vmin=-abs(var.var_1_matrix_).max(),
#         vmax=abs(var.var_1_matrix_).max(),
#         save_path="../reports/figures/annual/{}/heatmap_VAR1_matrix_OLS.pdf".format(
#             year
#         ),
#     )
#     var_cv = var.fit_elastic_net_cv(
#         grid=var_grid,
#         return_cv=True,
#         weighting_method=weighting_method,
#         penalise_diagonal=True,
#     )
#     return (var_cv, var)


# def estimate_cov(data, cov_grid):
#     cov_cv = GridSearchCV(
#         euraculus.covar.AdaptiveThresholdEstimator(),
#         param_grid=cov_grid,
#         cv=12,
#         n_jobs=-1,
#         verbose=1,
#     ).fit(data)
#     cov = cov_cv.best_estimator_
#     return (cov_cv, cov)


### PLOTS ###############################


# def create_data_plots(year, preprocessed_data, df_volas, df_spy_vola):
#     euraculus.plot.corr_heatmap(
#         df_volas.corr(),
#         title="Data Correlation prior to Decomposition",
#         save_path="../reports/figures/annual/{}/heatmap_total_variance_correlation.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.corr_heatmap(
#         preprocessed_data.corr(),
#         title="Data Correlation",
#         save_path="../reports/figures/annual/{}/heatmap_data_correlation.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.corr_heatmap(
#         euraculus.utils.autocorrcoef(preprocessed_data, lag=1),
#         title="Data Auto-Correlation (First order)",
#         save_path="../reports/figures/annual/{}/heatmap_data_autocorrelation.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.histogram(
#         preprocessed_data.stack(),
#         title="Distribution of Processed Data",
#         save_path="../reports/figures/annual/{}/histogram_data.pdf".format(year),
#     )
#     plt.close("all")


# def create_var_plots(year, var_cv, var):
#     euraculus.plot.net_cv_contour(
#         var_cv,
#         12,
#         logy=True,
#         save_path="../reports/figures/annual/{}/contour_VAR.pdf".format(year),
#     )
#     euraculus.plot.corr_heatmap(
#         var.var_1_matrix_,
#         title="Regularized VAR(1) coefficient matrix (Adaptive Elastic Net)",
#         vmin=-abs(var.var_1_matrix_).max(),
#         vmax=abs(var.var_1_matrix_).max(),
#         save_path="../reports/figures/annual/{}/heatmap_VAR1_matrix.pdf".format(year),
#     )
#     euraculus.plot.corr_heatmap(
#         pd.DataFrame(var.residuals_).corr(),
#         title="VAR Residual Correlation",
#         save_path="../reports/figures/annual/{}/heatmap_VAR_residual_correlation.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.corr_heatmap(
#         euraculus.utils.autocorrcoef(pd.DataFrame(var.residuals_), lag=1),
#         title="VAR Residual Auto-Correlation (First order)",
#         save_path="../reports/figures/annual/{}/heatmap_VAR_residual_autocorrelation.pdf".format(
#             year
#         ),
#     )
#     res_cov = pd.DataFrame(var.residuals_).cov()
#     euraculus.plot.corr_heatmap(
#         res_cov,
#         title="VAR Residual Sample Covariance Matrix",
#         vmin=-abs(res_cov.values).max(),
#         vmax=abs(res_cov.values).max(),
#         save_path="../reports/figures/annual/{}/heatmap_VAR_residual_covariance.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.histogram(
#         pd.DataFrame(var.residuals_).stack(),
#         title="Distribution of VAR Residuals",
#         save_path="../reports/figures/annual/{}/histogram_VAR_residuals.pdf".format(
#             year
#         ),
#     )
#     plt.close("all")


# def create_cov_plots(year, cov_cv, cov):
#     euraculus.plot.cov_cv_contour(
#         cov_cv,
#         12,
#         logy=False,
#         save_path="../reports/figures/annual/{}/contour_cov.pdf".format(year),
#     )
#     euraculus.plot.corr_heatmap(
#         cov.covar_,
#         title="Adaptive Threshold Estimate of VAR Residual Covariances",
#         vmin=-abs(cov.covar_).max(),
#         vmax=abs(cov.covar_).max(),
#         save_path="../reports/figures/annual/{}/heatmap_cov_matrix.pdf".format(year),
#     )
#     plt.close("all")


# def create_fevd_plots(year, fevd, horizon, column_to_ticker):
#     euraculus.plot.corr_heatmap(
#         pd.DataFrame(fevd.fev_single(horizon))
#         - np.diag(np.diag(fevd.fev_single(horizon))),
#         "FEV Single Contributions",
#         vmin=0,
#         cmap="binary",
#         infer_vmax=True,
#         save_path="../reports/figures/annual/{}/heatmap_FEV_contributions.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.corr_heatmap(
#         pd.DataFrame(fevd.decompose_fev(horizon=horizon, normalise=False))
#         - np.diag(np.diag(fevd.decompose_fev(horizon=horizon, normalise=False))),
#         "FEV Decomposition",
#         vmin=0,
#         vmax=None,
#         cmap="binary",
#         save_path="../reports/figures/annual/{}/heatmap_FEV_decomposition.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.corr_heatmap(
#         pd.DataFrame(fevd.decompose_fev(horizon=horizon, normalise=True))
#         - np.diag(np.diag(fevd.decompose_fev(horizon=horizon, normalise=True))),
#         "FEV Decomposition (row-normalised)",
#         vmin=0,
#         vmax=None,
#         cmap="binary",
#         save_path="../reports/figures/annual/{}/heatmap_FEV_decomposition_normalised.pdf".format(
#             year
#         ),
#     )

#     euraculus.plot.corr_heatmap(
#         pd.DataFrame(fevd.fu_single(horizon))
#         - np.diag(np.diag(fevd.fu_single(horizon))),
#         "FU Single Contributions",
#         vmin=0,
#         cmap="binary",
#         infer_vmax=True,
#         save_path="../reports/figures/annual/{}/heatmap_FU_contributions.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.corr_heatmap(
#         pd.DataFrame(fevd.decompose_fu(horizon=horizon, normalise=False))
#         - np.diag(np.diag(fevd.decompose_fu(horizon=horizon, normalise=False))),
#         "FU Decomposition",
#         vmin=0,
#         vmax=None,
#         cmap="binary",
#         save_path="../reports/figures/annual/{}/heatmap_FU_decomposition.pdf".format(
#             year
#         ),
#     )
#     euraculus.plot.corr_heatmap(
#         pd.DataFrame(fevd.decompose_fu(horizon=horizon, normalise=True))
#         - np.diag(np.diag(fevd.decompose_fu(horizon=horizon, normalise=True))),
#         "FU Decomposition (row-normalised)",
#         vmin=0,
#         vmax=None,
#         cmap="binary",
#         save_path="../reports/figures/annual/{}/heatmap_FU_decomposition_normalised.pdf".format(
#             year
#         ),
#     )

#     euraculus.plot.network_graph(
#         fevd.to_fev_graph(horizon, normalise=False),
#         column_to_ticker,
#         title="FEVD Network (FEV absolute)",
#         red_percent=5,
#         linewidth=0.25,
#         save_path="../reports/figures/annual/{}/network_FEV_absolute.png".format(year),
#     )
#     euraculus.plot.network_graph(
#         fevd.to_fev_graph(horizon, normalise=True),
#         column_to_ticker,
#         title="FEVD Network (FEV %)",
#         red_percent=2,
#         linewidth=0.25,
#         save_path="../reports/figures/annual/{}/network_FEV_normalised.png".format(
#             year
#         ),
#     )

#     euraculus.plot.network_graph(
#         fevd.to_fu_graph(horizon, normalise=False),
#         column_to_ticker,
#         title="FEVD Network (FU absolute)",
#         red_percent=5,
#         linewidth=0.25,
#         save_path="../reports/figures/annual/{}/network_FU_absolute.png".format(year),
#     )
#     euraculus.plot.network_graph(
#         fevd.to_fu_graph(horizon, normalise=True),
#         column_to_ticker,
#         title="FEVD Network (FU %)",
#         red_percent=2,
#         linewidth=0.25,
#         save_path="../reports/figures/annual/{}/network_FU_normalised.png".format(year),
#     )
#     plt.close("all")
