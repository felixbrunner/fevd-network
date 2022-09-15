# -*- coding: utf-8 -*-
# %% [markdown]
# # Regularized Factor FEVD estimation - Single window
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import warnings
import datetime as dt

import kungfu as kf
import matplotlib.pyplot as plt
import missingno as mno
import networkx as nx
import numpy as np

import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from euraculus.data import DataMap
from euraculus.var import FactorVAR
from euraculus.covar import AdaptiveThresholdEstimator, GLASSO
from euraculus.fevd import FEVD
from euraculus.estimate import (
    load_estimation_data,
    construct_pca_factors,
    build_lookup_table,
    estimate_fevd,
)
from euraculus.utils import (
    autocorrcoef,
    prec_to_pcorr,
)
from euraculus.plot import (
    corr_heatmap,
    missing_data,
    histogram,
    var_timeseries,
    net_cv_contour,
    net_scatter_losses,
    # network_graph,
    draw_fevd_as_network,
    cov_cv_contour,
    cov_scatter_losses,
    plot_glasso_cv,
    plot_pca_elbow,
    plot_size_concentration,
)
from euraculus.settings import (
    DATA_DIR,
    FACTORS,
    VAR_GRID,
    COV_GRID,
    HORIZON,
)

# %% [markdown]
# ## Setup
# ### Parameters

# %%
sampling_date = dt.datetime(year=2021, month=12, day=31)

# %%
save_outputs = True

# %% [markdown]
# ### Load data

# %%
data = DataMap(DATA_DIR)

# %%
# %%time
df_info, df_log_mcap_vola, df_factors = load_estimation_data(
    data=data, sampling_date=sampling_date
)
df_pca = construct_pca_factors(df=df_log_mcap_vola, n_factors=2)
df_factors = df_factors.merge(df_pca, right_index=True, left_index=True)
df_summary = data.load_selection_summary(sampling_date=sampling_date)

# %% [markdown]
# Make and export ticker table:

# %%
df_tickers = build_lookup_table(df_info)

# store
if save_outputs:
    kf.frame.FinancialDataFrame(df_tickers).export_to_latex(
        filename="tickers.tex", path=f"../reports/{sampling_date.date()}/"
    )
    data.dump(data=df_tickers, path=f"../reports/{sampling_date.date()}/tickers.csv")

# %% [markdown]
# ## Load estimates / Estimate

# %%
reestimate = False

# %%
# %%time
var_data = df_log_mcap_vola
factor_data = df_factors[FACTORS]

if reestimate:
    var_cv, var, cov_cv, cov, fevd = estimate_fevd(
        var_data=var_data,
        factor_data=factor_data,
        var_grid=VAR_GRID,
        cov_grid=COV_GRID,
    )

else:
    # try to read existing estimates
    try:
        var_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var_cv.pkl")
        var = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/var.pkl")
        cov_cv = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov_cv.pkl")
        cov = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/cov.pkl")
        fevd = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")
        residuals = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/residuals.pkl")
        warnings.warn(
            "using estimates from disc, model might not match with specification"
        )

    # estimate if no estimates are found
    except ValueError:
        var_cv, var, cov_cv, cov, fevd = estimate_fevd(
            var_data=var_data,
            factor_data=factor_data,
            var_grid=VAR_GRID,
            cov_grid=COV_GRID,
        )

# %% [markdown]
# ## Analysis
# ### Data

# %%
plot_size_concentration(
    df_summary,
    sampling_date=sampling_date,
    save_path=f"../reports/{sampling_date.date()}/data/value_concentration.pdf"
    if save_outputs
    else None,
)

# %%
histogram(
    np.exp(df_log_mcap_vola).fillna(0).stack(),
    title="Distribution of Intraday Value Volatilities",
    save_path=f"../reports/{sampling_date.date()}/data/histogram_value_volatility.pdf"
    if save_outputs
    else None,
    drop_tails=0.01,
    bins=100,
)

# %%
histogram(
    df_log_mcap_vola.stack(),
    title="Distribution of Log Intraday Value Volatilities",
    save_path=f"../reports/{sampling_date.date()}/data/histogram_log_value_volatility.pdf"
    if save_outputs
    else None,
)

# %%
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        idx = i * 3 + j
        ax = axes[i, j]
        ax.hist(np.exp(df_log_mcap_vola).iloc[:, idx], bins=30)
        ax.set_title(df_log_mcap_vola.columns[idx])

# %%
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        idx = i * 3 + j
        ax = axes[i, j]
        ax.hist(df_log_mcap_vola.iloc[:, idx], bins=30)
        ax.set_title(df_log_mcap_vola.columns[idx])

# %%
missing_data(
    df_log_mcap_vola.replace(0, np.nan),
    title="Missing Data: Logged Dollar Volatility",
    save_path=f"../reports/{sampling_date.date()}/data/matrix_missing_data.pdf"
    if save_outputs
    else None,
)


# %%
def reorder_matrix(
    df: np.ndarray, df_info: pd.DataFrame, sorting_columns: list, index_columns: list
) -> pd.DataFrame:
    """"""
    ordered_info = df_info.reset_index().sort_values(sorting_columns, ascending=False)
    ordered_row_indices = ordered_info.index.values.tolist()
    ordered_index = ordered_info.set_index(index_columns).index
    df_ordered = pd.DataFrame(df).iloc[ordered_row_indices, ordered_row_indices]
    return df_ordered, ordered_index


# %%
matrix, index = reorder_matrix(
    df_log_mcap_vola.corr(),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Total Valuation Volatility Correlations",
    save_path=f"../reports/{sampling_date.date()}/data/heatmap_total_correlation.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    autocorrcoef(df_log_mcap_vola, lag=1),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Total Value Volatility Auto-Correlations (First order)",
    save_path=f"../reports/{sampling_date.date()}/data/heatmap_total_autocorrelation.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
pd.Series(np.diag(autocorrcoef(df_log_mcap_vola, lag=1))).plot(
    kind="hist", title="Diagonal Autocorrelations", bins=20
)
plt.show()

# %% [markdown]
# ### Factors

# %%
plot_pca_elbow(
    df_log_mcap_vola,
    n_pcs=10,
    save_path=f"../reports/{sampling_date.date()}/data/lineplot_pca_explained_variance.pdf"
    if save_outputs
    else None,
)

# %%
df_factors.corr()

# %%
corr_heatmap(
    df_factors.corr(),
    title="Factor correlations",
    save_path=f"../reports/{sampling_date.date()}/data/heatmap_factor_correlation.pdf"
    if save_outputs
    else None,
    labels=df_factors.columns,
)

# %%
df_factors.plot(kind="hist", bins=100, alpha=0.5)

# %%
df_factors.plot()

# %% [markdown]
# ### VAR

# %% [markdown]
# $s_{i,t} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} f_{k,t} + \sum_{j=1}^{N} \gamma_{i,j} s_{j,t-1}  + u_{i,t}$
#
# where
# - $s_{i,t}$: $1 \times 1$ (asset value volatility)
# - $\alpha_i$: $1 \times 1$ (intercepts/pricing errors)
# - $f_{k,t}$: $1 \times 1$ (factor excess returns)
# - $\beta_{i,k}$: $1 \times 1$ (factor loadings)
# - $\gamma_{i,j}$: $1 \times 1$ (VAR coefficients)
# - $u_{i,t}$: $1 \times 1$ (error term)
#
# Adaptive Elastic Net with hyperparameters $\lambda, \kappa$
#
# $(\hat{\alpha}, \hat{\beta}, \hat{\gamma})= \underset{(\alpha, \beta, \gamma)}{argmin} \Bigg[ \frac{1}{2NT}\sum_{i=1}^{N}\sum_{t=1}^{T} \Big(s_{i,t} - \big(\alpha_i + \sum_{k=1}^{K} \beta_{i,k} f_{k,t} + \sum_{j=1}^{N} \gamma_{i,j} s_{j,t-1}\big)\Big)^2 + \lambda \sum_{i=1}^{N}\sum_{j=1}^{N} w_{i,j} \big(\kappa |\gamma_{i,j}| + (1-\kappa) \frac{1}{2} \gamma_{i,j}^2\big)\Bigg]$
#
# weights are set to $w_{i,j} =|\hat{\beta}_{i,j}^{Ridge}|^{-1}$

# %%
ols_var = FactorVAR(has_intercepts=True, p_lags=1)
ols_var.fit(var_data=var_data, factor_data=factor_data, method="OLS")

# %%
matrix, index = reorder_matrix(
    ols_var.var_1_matrix_,
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Non-regularized VAR(1) coefficients (OLS)",
    infer_limits=True,
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_ols_var1_matrix.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
print("OLS ESTIMATE")
print(
    "total R2: {:.2f}%\nfactor R2: {:.2f}%".format(
        ols_var.r2(var_data=var_data, factor_data=factor_data) * 100,
        ols_var.factor_r2(var_data=var_data, factor_data=factor_data) * 100,
    )
)
print("Partial R2: ", ols_var.partial_r2s(var_data=var_data, factor_data=factor_data))
print(
    "Component R2: ", ols_var.component_r2s(var_data=var_data, factor_data=factor_data)
)
print("VAR(1) matrix is {:.2f}% dense.".format(ols_var.var_density_ * 100))

# %%
net_cv_contour(
    var_cv,
    15,
    logx=True,
    logy=True,
    save_path=f"../reports/{sampling_date.date()}/regression/contour_var.pdf"
    if save_outputs
    else None,
)

# %%
print("AENET ESTIMATE")
print(
    "total R2: {:.2f}%\nfactor R2: {:.2f}%".format(
        var.r2(var_data=var_data, factor_data=factor_data) * 100,
        var.factor_r2(var_data=var_data, factor_data=factor_data) * 100,
    )
)
print("Partial R2: ", var.partial_r2s(var_data=var_data, factor_data=factor_data))
print("Component R2: ", var.component_r2s(var_data=var_data, factor_data=factor_data))
print("VAR(1) matrix is {:.2f}% dense.".format(var.var_density_ * 100))

# %%
net_scatter_losses(
    var_cv,
    save_path=f"../reports/{sampling_date.date()}/regression/scatter_var.pdf"
    if save_outputs
    else None,
)

# %%
matrix, index = reorder_matrix(
    var.var_1_matrix_,
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Regularized VAR(1) coefficients (Adaptive Elastic Net)",
    infer_limits=True,
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_aenet_var1_matrix.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
factor_residuals = var.factor_residuals(var_data=var_data, factor_data=factor_data)

# %%
histogram(
    factor_residuals.stack(),
    title="Distribution of VAR Factor Residuals",
    save_path=f"../reports/{sampling_date.date()}/regression/histogram_VAR_factor_residuals.pdf"
    if save_outputs
    else None,
)

# %%
matrix, index = reorder_matrix(
    factor_residuals.corr(),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="VAR Factor Residual Correlation",
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_VAR_factor_residual_correlation.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    autocorrcoef(factor_residuals, lag=1),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="VAR Factor Residual Auto-Correlation (First order)",
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_VAR_factor_residual_autocorrelation.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
residuals = var.residuals(var_data=var_data, factor_data=factor_data)

# %%
histogram(
    residuals.stack(),
    title="Distribution of VAR Residuals",
    save_path=f"../reports/{sampling_date.date()}/regression/histogram_VAR_residuals.pdf"
    if save_outputs
    else None,
)

# %%
matrix, index = reorder_matrix(
    residuals.corr(),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="VAR Residual Correlation",
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_VAR_residual_correlation.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    autocorrcoef(residuals, lag=1),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="VAR Residual Auto-Correlation (First order)",
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_VAR_residual_autocorrelation.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    autocorrcoef(residuals, lag=2),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="VAR Residual Auto-Correlation (Second order)",
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_VAR_residual_autocorrelation_2nd.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    autocorrcoef(residuals, lag=3),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="VAR Residual Auto-Correlation (Third order)",
    save_path=f"../reports/{sampling_date.date()}/regression/heatmap_VAR_residual_autocorrelation_3rd.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %% [markdown]
# ### Covariance matrix

# %%
matrix, index = reorder_matrix(
    residuals.cov(),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Sample Estimate of the VAR Residual Covariance Matrix",
    infer_limits=True,
    save_path=f"../reports/{sampling_date.date()}/covariance/heatmap_VAR_residual_covariance.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    prec_to_pcorr(np.linalg.inv(residuals.cov())),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Sample Estimate of the VAR Residual Partial Correlation Matrix",
    save_path=f"../reports/{sampling_date.date()}/covariance/heatmap_VAR_residual_partial_corr.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
plot_glasso_cv(
    cov_cv,
    save_path=f"../reports/{sampling_date.date()}/covariance/line_cov_cv.pdf"
    if save_outputs
    else None,
)

# %%
matrix, index = reorder_matrix(
    cov.covariance_,
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
lim = residuals.cov().abs().values.max()
corr_heatmap(
    matrix,
    title="Graphical Lasso Estimate of VAR Residual Covariance Matrix",
    vmin=-lim,
    vmax=lim,
    save_path=f"../reports/{sampling_date.date()}/covariance/heatmap_cov_matrix.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    cov.precision_,
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    infer_limits=True,
    title="Graphical Lasso Estimate of VAR Residual Precision Matrix",
    save_path=f"../reports/{sampling_date.date()}/covariance/heatmap_precision_matrix.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    prec_to_pcorr(cov.precision_),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Graphical Lasso Estimate of VAR Residual Partial Correlation Matrix",
    save_path=f"../reports/{sampling_date.date()}/covariance/heatmap_partial_corr_matrix.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
print(f"Precision matrix is {cov.precision_density_*100:.2f}% dense")

# %% [markdown]
# ### FEVD

# %%
stat, pval = fevd.test_diagonal_generalized_innovations(t_observations=len(var_data))
outcome = "Reject" if pval < 0.05 else "Do not reject"
print(
    f"{outcome} H0: independent generalized innovations at 5% confidence with test statistic {stat:.2f} and p-value {pval:.4f}"
)

# %%
matrix, index = reorder_matrix(
    cov.covariance_ @ np.diag(np.diag(cov.covariance_) ** -0.5),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Generalized Impulse Matrix",
    infer_limits=True,
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_generalized_impulse_matrix.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    sp.linalg.sqrtm(cov.covariance_),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="Matrix Square Root Impulse Matrix",
    infer_limits=True,
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_sqrtm_impulse_matrix.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %% [markdown]
# #### VMA Matrices

# %%
# lim = abs(fevd.vma_matrix(1)).max()

# %%
for h in [0, 1, 2, 3, 5, 10, 21]:
    matrix, index = reorder_matrix(
        fevd.vma_matrix(h),
        df_info,
        ["ff_sector_ticker", "mean_valuation_volatility"],
        ["ff_sector_ticker", "ticker"],
    )
    primary_labels = index.get_level_values(1).tolist()
    secondary_labels = index.get_level_values(0).tolist()
    corr_heatmap(
        matrix,
        title=f"VMA({h}) Matrix",
        infer_limits=True,  # , vmin=-lim, vmax=lim,
        save_path=f"../reports/{sampling_date.date()}/vma/heatmap_vma{h}_matrix.pdf"
        if save_outputs
        else None,
        labels=primary_labels,
        secondary_labels=secondary_labels,
    )
    plt.show()

# %% [markdown]
# #### Impulse Response Matrices

# %%
for h in [0, 1, 2, 3, 5, 10, 21]:
    matrix, index = reorder_matrix(
        fevd.impulse_response_functions(h),
        df_info,
        ["ff_sector_ticker", "mean_valuation_volatility"],
        ["ff_sector_ticker", "ticker"],
    )
    primary_labels = index.get_level_values(1).tolist()
    secondary_labels = index.get_level_values(0).tolist()
    corr_heatmap(
        matrix,
        title=f"Impulse Response({h}) Matrix",
        infer_limits=True,
        save_path=f"../reports/{sampling_date.date()}/irf/heatmap_ir{h}_matrix.pdf"
        if save_outputs
        else None,
        labels=primary_labels,
        secondary_labels=secondary_labels,
    )
    plt.show()

# %% [markdown]
# #### Innovation Response Variance Matrices

# %%
for h in [0, 1, 2, 3, 5, 10, 21]:
    matrix, index = reorder_matrix(
        fevd.innovation_response_variances(h),
        df_info,
        ["ff_sector_ticker", "mean_valuation_volatility"],
        ["ff_sector_ticker", "ticker"],
    )
    primary_labels = index.get_level_values(1).tolist()
    secondary_labels = index.get_level_values(0).tolist()
    corr_heatmap(
        matrix,
        title=f"Innovation Response Variance ({h}) Matrix",
        infer_limits=True,
        save_path=f"../reports/{sampling_date.date()}/irv/heatmap_irv{h}_matrix.pdf"
        if save_outputs
        else None,
        labels=primary_labels,
        secondary_labels=secondary_labels,
    )
    plt.show()

# %% [markdown]
# #### FEV Adjacency

# %%
matrix, index = reorder_matrix(
    pd.DataFrame(fevd.forecast_error_variances(HORIZON))
    - np.diag(np.diag(fevd.forecast_error_variances(HORIZON))),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="FEV Single Contributions",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_FEV_contributions.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    pd.DataFrame(
        fevd.forecast_error_variance_decomposition(horizon=HORIZON, normalize=False)
    )
    - np.diag(
        np.diag(
            fevd.forecast_error_variance_decomposition(horizon=HORIZON, normalize=False)
        )
    ),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="FEV Decomposition",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_FEV_decomposition.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    pd.DataFrame(
        fevd.forecast_error_variance_decomposition(horizon=HORIZON, normalize=True)
    )
    - np.diag(
        np.diag(
            fevd.forecast_error_variance_decomposition(horizon=HORIZON, normalize=True)
        )
    ),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="FEV Decomposition (row-normalised)",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_FEV_decomposition_normalised.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %% [markdown]
# #### FU Adjacency

# %%
matrix, index = reorder_matrix(
    pd.DataFrame(fevd.forecast_uncertainty(HORIZON))
    - np.diag(np.diag(fevd.forecast_uncertainty(HORIZON))),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="FU Single Contributions",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_FU_contributions.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    pd.DataFrame(
        fevd.forecast_uncertainty_decomposition(horizon=HORIZON, normalize=False)
    )
    - np.diag(
        np.diag(
            fevd.forecast_uncertainty_decomposition(horizon=HORIZON, normalize=False)
        )
    ),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="FU Decomposition",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_FU_decomposition.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %%
matrix, index = reorder_matrix(
    pd.DataFrame(
        fevd.forecast_uncertainty_decomposition(horizon=HORIZON, normalize=True)
    )
    - np.diag(
        np.diag(
            fevd.forecast_uncertainty_decomposition(horizon=HORIZON, normalize=True)
        )
    ),
    df_info,
    ["ff_sector_ticker", "mean_valuation_volatility"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
corr_heatmap(
    matrix,
    title="FU Decomposition (row-normalised)",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path=f"../reports/{sampling_date.date()}/network/heatmap_FU_decomposition_normalised.pdf"
    if save_outputs
    else None,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)

# %% [markdown]
# ### Network structure

# %%
_ = draw_fevd_as_network(
    fevd,
    df_info,
    horizon=HORIZON,
    table_name="fev",
    normalize=False,
    title="FEV Network",
    save_path=f"../reports/{sampling_date.date()}/network/network_FEV.png"
    if save_outputs
    else None,
)

# %%
_ = draw_fevd_as_network(
    fevd,
    df_info,
    horizon=HORIZON,
    table_name="fevd",
    normalize=False,
    title="FEVD Network",
    save_path=f"../reports/{sampling_date.date()}/network/network_FEVD.png"
    if save_outputs
    else None,
)

# %%
# ticker lookup
data.lookup_ticker(
    tickers=["HON", "CMCSA", "CHTR", "SNOW", "ZM", "SQ", "MU"], date=sampling_date
)


# %% [markdown]
# ### Contributions

# %%
def plot_contribution_bars(
    scores: np.ndarray, names: list, title: str = None, normalize: bool = False
):
    """"""
    # prepare data
    data = pd.Series(scores, index=names).sort_values(ascending=False)
    if normalize:
        data /= data.sum()

    # plot
    fix, ax = plt.subplots(1, 1, figsize=(17, 5))
    ax.bar(x=np.arange(1, len(data) + 1), height=data)
    ax.set_title(title + (" (normalized to one)" if normalize else ""))

    # format
    ax.set_xlim([0, 101])
    ax.set_xticks(np.arange(1, len(scores) + 1), minor=False)
    ax.set_xticklabels(names, rotation=90, minor=False)
    ax.tick_params(axis="x", which="major", bottom=True, labelbottom=True)


# %%
plot_contribution_bars(
    scores=df_info["mean_size"],
    names=var_data.columns,
    title="Market capitalization",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=np.exp(df_log_mcap_vola).mean().values,
    names=var_data.columns,
    title="Valuation Volatility",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.amplification_factor(21, table_name="fev").flatten(),
    names=var_data.columns,
    title="Amplification Factor",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.absorption_rate(21, table_name="fev").flatten(),
    names=var_data.columns,
    title="Absorption Rate",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.in_connectedness(21, table_name="fev").flatten(),
    names=var_data.columns,
    title="In-Connectedness",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.out_connectedness(21, table_name="fev").flatten(),
    names=var_data.columns,
    title="Out-Connectedness",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.in_eigenvector_centrality(21, table_name="fev").flatten(),
    names=var_data.columns,
    title="In-Eigenvector Centrality",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.out_eigenvector_centrality(21, table_name="fev").flatten(),
    names=var_data.columns,
    title="Out-Eigenvector Centrality",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.in_page_rank(
        21,
        table_name="fev",
        weights=np.exp(df_log_mcap_vola).mean().values.reshape(-1, 1),
        alpha=0.85,
    ).flatten(),
    names=var_data.columns,
    title="In-Page Rank (α=0.85)",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.out_page_rank(
        21,
        table_name="fev",
        weights=np.exp(df_log_mcap_vola).mean().values.reshape(-1, 1),
        alpha=0.85,
    ).flatten(),
    names=var_data.columns,
    title="Out-Page Rank (α=0.85)",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.in_page_rank(
        21,
        table_name="fev",
        weights=np.exp(df_log_mcap_vola).mean().values.reshape(-1, 1),
        alpha=0.95,
    ).flatten(),
    names=var_data.columns,
    title="In-Page Rank (α=0.95)",
    normalize=False,
)

# %%
plot_contribution_bars(
    scores=fevd.out_page_rank(
        21,
        table_name="fev",
        weights=np.exp(df_log_mcap_vola).mean().values.reshape(-1, 1),
        alpha=0.95,
    ).flatten(),
    names=var_data.columns,
    title="Out-Page Rank (α=0.95)",
    normalize=False,
)
