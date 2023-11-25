# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

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

from euraculus.data.map import DataMap
from euraculus.models.var import FactorVAR
from euraculus.models.covariance import GLASSO
from euraculus.network.fevd import FEVD
from euraculus.models.estimate import (
    load_estimation_data,
    build_lookup_table,
    estimate_fevd,
)
from euraculus.utils.utils import (
    autocorrcoef,
    prec_to_pcorr,
)
from euraculus.utils.plot import (
    distribution_plot,
    save_ax_as_pdf,
    missing_data_matrix,
    matrix_heatmap,
    draw_network,
    contribution_bars,
)
from euraculus.settings import (
    OUTPUT_DIR,
    DATA_DIR,
    FACTORS,
    VAR_GRID,
    COV_GRID,
    HORIZON,
    SECTOR_COLORS,
)

# %% [markdown]
# ## Setup
# ### Parameters

# %%
sampling_date = dt.datetime(year=2022, month=3, day=31)
# sampling_date = dt.datetime(year=2022, month=12, day=31)
# sampling_date = dt.datetime(year=1993, month=5, day=31)

# %%
save_outputs = True

# %% [markdown]
# ### Load data

# %%
data = DataMap(DATA_DIR)

# %%
# %%time
df_info, df_log_vola, df_factors = load_estimation_data(
    data=data, sampling_date=sampling_date
)
# df_pca = construct_pca_factors(df=df_log_vola, n_factors=2)
# df_factors = df_factors.merge(df_pca, right_index=True, left_index=True)
df_summary = data.load_selection_summary(sampling_date=sampling_date)

# %% [markdown]
# Make and export ticker table:

# %%
df_tickers = build_lookup_table(df_info)

# store
if save_outputs:
    kf.frame.FinancialDataFrame(df_tickers).export_to_latex(
        filename="tickers.tex", path=str(OUTPUT_DIR / f"{sampling_date.date()}/")
    )
    data.dump(data=df_tickers, path=str(OUTPUT_DIR / f"{sampling_date.date()}/tickers.csv"))

# %%
df_tickers

# %%
df_hist = data.load_historic(sampling_date=sampling_date)

# %%
df_hist["anndummy"].unstack().sum().value_counts()

# %%
doubledumm = df_hist["anndummy"].unstack().replace(0, np.nan).ffill(limit=1).replace(np.nan, 0).stack()

# %%
annret = (df_hist["retadj"] * doubledumm).groupby("date").mean()

# %%
annres = (df_hist["capm_alpharesid"] * doubledumm).groupby("date").mean()

# %%
annret.plot()

# %%
annres.plot()

# %%
annret.to_frame().join(annres.rename("res")).std() # * np.sqrt(250)

# %%
annret.to_frame().join(annres.rename("res")).corr()

# %% [markdown]
# ## Load estimates / Estimate

# %%
reestimate = False

# %%
# %%time
var_data = df_log_vola
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

# %% [markdown]
# ### Plot concentration of MCap

# %%
# create plot
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax2 = ax.twinx()
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ax.set_title(f"Firm Size Distribution ({sampling_date.date()})")

# prepare data
mcaps = df_summary["last_mcap"] * 1e3
mcaps = mcaps.loc[mcaps > 0].sort_values(ascending=False).reset_index(drop=True)
mcaps.index = mcaps.index+1
cumulative = pd.Series([0]).append(mcaps.cumsum()) / mcaps.sum()

# firm sizes
area = ax.fill_between(
    x=mcaps.index,
    y1=mcaps,
    # y2=1e0,
    label="Asset market capitalization",
    alpha=0.25,
    # linewidth=1,
    # edgecolor="k",
    # hatch="|",
)
ax.scatter(
    mcaps.index,
    mcaps,
    marker=".",
    color=colors[0],
    s=5,
)
scat = ax.scatter(
    x=df_summary.loc[var_data.columns, "last_mcap_rank"].values,
    y=df_summary.loc[var_data.columns, "last_mcap"].values * 1e3,
    label="100 sampled assets",
    marker="x",
    color=colors[1],
)

ax.set_ylabel("Market capitalization")  # ('000 USD)")
ax.set_xlabel("Size Rank")
# ax.set_ylim([0, mcaps.max()*1.05])
ax.set_xlim([-10, len(mcaps) + 10])
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="-")

# cumulative
line = ax2.plot(cumulative, label="Cumulative share (right axis)", color=colors[2])
ax2.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
ax2.set_yticklabels([f"{int(tick*100)}%" for tick in ax2.get_yticks()])
ax2.grid(True, color="gray")
ax2.set_ylabel("Cumulative share")
ax2.set_ylim([0, 1.01])

# cutoffs
for pct in ax2.get_yticks()[1:-1]:
    x = cumulative[cumulative.gt(pct)].index[0]
    ax2.scatter(x=x, y=cumulative[x], marker="o", color=colors[2])
    ax2.text(
        x=x + 10,
        y=cumulative[x] - 0.04,
        s=f"{x} assets: {cumulative[x]*100:.2f}% of total market capitalization",
        color=colors[2],
    )

# legend
elements = [area, scat, line[0]]
labels = [e.get_label() for e in elements]
ax.legend(elements, labels)  # , bbox_to_anchor=(1.05, 0.5), loc="center left")

# save
if save_outputs:
    fig.savefig(
        OUTPUT_DIR / f"{sampling_date.date()}/data/value_concentration.pdf",
        format="pdf",
        dpi=200,
        bbox_inches="tight",
    )

# %%
cumulative[10]

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax = distribution_plot(
    np.exp(var_data).fillna(0).stack(),
    drop_tails=0.01,
    title="Distribution of Intraday Volatilities",
)
if save_outputs:
    save_ax_as_pdf(
        ax, save_path=OUTPUT_DIR / f"{sampling_date.date()}/data/histogram_volatility.pdf"
    )

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax = distribution_plot(
    var_data.stack(),
    title="Distribution of Log Intraday Volatilities",
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/data/histogram_log_volatility.pdf",
    )

# %%
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        idx = i * 3 + j
        ax = axes[i, j]
        ax.hist(np.exp(var_data).iloc[:, idx], bins=30)
        ax.set_title(var_data.columns[idx])

# %%
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i in range(10):
    for j in range(10):
        idx = i * 3 + j
        ax = axes[i, j]
        ax.hist(var_data.iloc[:, idx], bins=30)
        ax.set_title(var_data.columns[idx])

# %%
ax = missing_data_matrix(
    var_data.replace(0, np.nan),
    title="Missing Data: Logged Dollar Volatility",
)
if save_outputs:
    save_ax_as_pdf(
        ax, save_path=OUTPUT_DIR / f"{sampling_date.date()}/data/matrix_missing_data.pdf"
    )


# %%
def reorder_matrix(
    df: np.ndarray, df_info: pd.DataFrame, sorting_columns: list, index_columns: list
) -> pd.DataFrame:
    """Reorder the rows and columns of a square matrix.

    Args:
        df:
        df_info:
        sorting_columns:
        index_columns:

    Returns:
        df_ordered:
        ordered_index:
    """
    ordered_info = df_info.reset_index().sort_values(sorting_columns, ascending=False)
    ordered_row_indices = ordered_info.index.values.tolist()
    ordered_index = ordered_info.set_index(index_columns).index
    df_ordered = pd.DataFrame(df).iloc[ordered_row_indices, ordered_row_indices]
    return df_ordered, ordered_index


# %%
matrix, index = reorder_matrix(
    var_data.corr(),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Total Valuation Volatility Correlations",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/data/heatmap_total_correlation.pdf",
    )

# %%
matrix, index = reorder_matrix(
    autocorrcoef(var_data, lag=1),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Total Value Volatility Auto-Correlations (First order)",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/data/heatmap_total_autocorrelation.pdf",
    )

# %%
pd.Series(np.diag(autocorrcoef(df_log_vola, lag=1))).plot(
    kind="hist", title="Diagonal Autocorrelations", bins=20
)
plt.show()

# %% [markdown]
# ### Factors

# %% [markdown]
# #### Plot PCA elbow

# %%
# parameters
n_pcs: int = 10

# create pca
pca = PCA(n_components=n_pcs).fit(var_data)

# plot data
fig, ax = plt.subplots(1, 1)
ax.plot(pca.explained_variance_ratio_, marker="o")

# xticks
ax.set_xticks(np.arange(n_pcs))
ax.set_xticklabels(np.arange(n_pcs) + 1)

# labels
ax.set_title("Principal Components: Explained Variance")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")

if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/data/lineplot_pca_explained_variance.pdf",
    )

# %%
df_corr = df_factors.join(pd.DataFrame(data=pca.transform(var_data), columns=[f"pc{i}" for i in range(1,11)], index=df_factors.index))

# %%
ax = matrix_heatmap(
    df_corr.corr(),
    title="Factor correlations",
    labels=df_corr.columns,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/data/heatmap_factor_correlation.pdf",
    )

# %%
df_factors[FACTORS].plot(kind="hist", bins=100, alpha=0.5)
plt.show()

# %%
df_factors[FACTORS].plot()
plt.show()

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
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Non-regularized VAR(1) coefficients (OLS)",
    infer_limits=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_ols_var1_matrix.pdf",
    )

# %%
print("OLS ESTIMATE")
print(
    f"R2: {ols_var.r2(var_data=var_data, factor_data=factor_data) * 100:.2f}"
)
print("Partial R2: ", ols_var.partial_r2s(var_data=var_data, factor_data=factor_data))
print(
    "Component R2: ", ols_var.component_r2s(var_data=var_data, factor_data=factor_data)
)
print("VAR(1) matrix is {:.2f}% dense.".format(ols_var.var_density_ * 100))

# %% [markdown]
# #### Elastic net CV contour plot

# %%
# parameters
levels: int = 15
logx: bool = True
logy: bool = True

# create plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_title("Adaptive Elastic Net Hyper-Parameter Search Grid")

# data
x_name, y_name = var_cv.param_grid.keys()
x_values, y_values = var_cv.param_grid.values()
x_grid, y_grid = np.meshgrid(x_values, y_values)
z_values = (
    -var_cv.cv_results_["mean_test_score"].reshape(len(x_values), len(y_values)).T
)

# contour plotting
contour = ax.contourf(
    x_grid,
    y_grid,
    z_values,
    levels=levels,
    cmap="RdYlGn_r",
    antialiased=True,
    alpha=1,
)
ax.contour(
    x_grid,
    y_grid,
    z_values,
    levels=levels,
    colors="k",
    antialiased=True,
    linewidths=1,
    alpha=0.6,
)
ax.contour(
    x_grid,
    y_grid,
    z_values,
    levels=[1.0],
    colors="k",
    antialiased=True,
    linewidths=2,
    alpha=1,
)
cb = fig.colorbar(contour)

# grid & best estimator
x_v = [a[x_name] for a in var_cv.cv_results_["params"]]
y_v = [a[y_name] for a in var_cv.cv_results_["params"]]
ax.scatter(x_v, y_v, marker=".", label="grid", color="k", alpha=0.25)
ax.scatter(
    *var_cv.best_params_.values(),
    label="best estimator",
    marker="x",
    s=150,
    color="k",
    zorder=2,
)

# labels & legend
ax.set_xlabel("$\kappa$ (0=ridge, 1=LASSO)")
ax.set_ylabel("$\lambda$ (0=OLS, $\infty$=zeros)")
ax.legend()  # loc='upper left')
cb.set_label("Cross-Validation MSE (Standardized data)", rotation=90)
v = (1 - cb.vmin) / (cb.vmax - cb.vmin)
cb.ax.plot([0, 1], [v, v], "k", linewidth=2)
if logx:
    ax.set_xscale("log")
if logy:
    ax.set_yscale("log")

# limits
ax.set_xlim([min(x_values), max(x_values)])
ax.set_ylim([min(y_values), max(y_values)])

if save_outputs:
    save_ax_as_pdf(
        ax, save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/contour_var.pdf"
    )

# %%
print("AENET ESTIMATE")
print(
    f"R2: {var.r2(var_data=var_data, factor_data=factor_data) * 100:.2f}"
)
print("Partial R2: ", var.partial_r2s(var_data=var_data, factor_data=factor_data))
print("Component R2: ", var.component_r2s(var_data=var_data, factor_data=factor_data))
print("VAR(1) matrix is {:.2f}% dense.".format(var.var_density_ * 100))

# %% [markdown]
# #### Elastic Net scatter losses

# %%
# extract data
train_losses = -var_cv.cv_results_["mean_train_score"]
valid_losses = -var_cv.cv_results_["mean_test_score"]
lambdas = pd.Series([d["lambdau"] for d in var_cv.cv_results_["params"]])
kappas = pd.Series([d["alpha"] for d in var_cv.cv_results_["params"]])
best = var_cv.best_index_

# figure parameters
fig, ax = plt.subplots(1, 1, figsize=(14,8))
colors = np.log(lambdas)
sizes = (np.log(kappas) + 12) * 20

# labels
ax.set_xlabel("Mean Training MSE (In-sample)")
ax.set_ylabel("Mean Validation MSE (Out-of-sample)")
ax.set_title("Adaptive Elastic Net Cross-Validation Errors")

# scatter plots
sc = ax.scatter(
    train_losses, valid_losses, c=colors, s=sizes, cmap="bone", edgecolor="k"
)
ax.scatter(
    train_losses[best],
    valid_losses[best],
    s=sizes[best] * 2,
    c="r",
    edgecolor="k",
    marker="x",
    zorder=100,
    label="best model",
)

# 45 degree line
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
ax.plot(lims, lims, color="grey", linestyle="--", label="45-degree line", zorder=0)

# legends
handles, _ = sc.legend_elements(prop="colors", num=colors.nunique())
color_legend = ax.legend(
    handles[2:],
    ["{:.1e}".format(i) for i in lambdas.unique()],
    loc="lower left",
    title="λ",
)
ax.add_artist(color_legend)

handles, _ = sc.legend_elements(prop="sizes", alpha=0.6, num=sizes.nunique())
size_legend = ax.legend(
    handles,
    ["{:.1e}".format(i) for i in kappas.unique()],
    loc="lower right",
    title="κ",
)
ax.add_artist(size_legend)
ax.legend(loc="lower center")

if save_outputs:
    save_ax_as_pdf(
        ax, save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/scatter_var.pdf"
    )

# %%
matrix, index = reorder_matrix(
    var.var_1_matrix_,
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Regularized VAR(1) coefficients (Adaptive Elastic Net)",
    infer_limits=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_aenet_var1_matrix.pdf",
    )

# %%
factor_residuals = var.factor_residuals(var_data=var_data, factor_data=factor_data)

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax = distribution_plot(
    factor_residuals.stack(),
    title="Distribution of VAR Factor Residuals",
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/histogram_VAR_factor_residuals.pdf",
    )

# %%
matrix, index = reorder_matrix(
    factor_residuals.corr(),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="VAR Factor Residual Correlation",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_VAR_factor_residual_correlation.pdf",
    )

# %%
matrix, index = reorder_matrix(
    autocorrcoef(factor_residuals, lag=1),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="VAR Factor Residual Auto-Correlation (First order)",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_VAR_factor_residual_autocorrelation.pdf",
    )

# %%
residuals = var.residuals(var_data=var_data, factor_data=factor_data)

# %%
fig, ax = plt.subplots(figsize=(10,6))
ax = distribution_plot(
    residuals.stack(),
    title="Distribution of VAR Residuals",
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/histogram_VAR_residuals.pdf",
    )

# %%
matrix, index = reorder_matrix(
    residuals.corr(),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="VAR Residual Correlation",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_VAR_residual_correlation.pdf",
    )

# %%
matrix, index = reorder_matrix(
    autocorrcoef(residuals, lag=1),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="VAR Residual Auto-Correlation (First order)",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_VAR_residual_autocorrelation.pdf",
    )

# %%
matrix, index = reorder_matrix(
    autocorrcoef(residuals, lag=2),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="VAR Residual Auto-Correlation (Second order)",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_VAR_residual_autocorrelation_2nd.pdf",
    )

# %%
matrix, index = reorder_matrix(
    autocorrcoef(residuals, lag=3),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="VAR Residual Auto-Correlation (Third order)",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/regression/heatmap_VAR_residual_autocorrelation_3rd.pdf",
    )

# %% [markdown]
# ### Covariance matrix

# %%
matrix, index = reorder_matrix(
    residuals.cov(),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Sample Estimate of the VAR Residual Covariance Matrix",
    infer_limits=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/covariance/heatmap_VAR_residual_covariance.pdf",
    )

# %%
matrix, index = reorder_matrix(
    prec_to_pcorr(np.linalg.inv(residuals.cov())),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Sample Estimate of the VAR Residual Partial Correlation Matrix",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/covariance/heatmap_VAR_residual_partial_corr.pdf",
    )

# %% [markdown]
# #### Plot GLASSO CV

# %%
# create plot
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xscale("log")
ax.set_xlabel("ρ (0=sample cov, $\infty = diag(\hat{\Sigma}$))")
ax.set_ylabel("Mean Cross-Validation Loss")
ax.set_title("Graphical Lasso Hyper-Parameter Search Grid")

# add elements
ax.plot(
    cov_cv.param_grid["alpha"],
    -cov_cv.cv_results_["mean_test_score"],
    marker="o",
    label="mean validation loss",
)
ax.plot(
    cov_cv.param_grid["alpha"],
    -cov_cv.cv_results_["mean_train_score"],
    marker="s",
    label="mean training loss",
    linestyle="--",
)# ax.axhline(-cov_cv.best_score_, label='Best Adaptive Threshold Estimate', linestyle=':', linewidth=1, color='k')

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ax.axvline(
    cov_cv.best_params_["alpha"],
    label="best estimator",
    color="k",
    linestyle=":",
    linewidth=1,
)
ax.scatter(
    cov_cv.best_params_["alpha"],
    -cov_cv.cv_results_["mean_test_score"][cov_cv.best_index_],
    color="k",
    marker="o",
    zorder=100,
    s=100,
)
ax.scatter(
    cov_cv.best_params_["alpha"],
    -cov_cv.cv_results_["mean_train_score"][cov_cv.best_index_],
    color="k",
    marker="s",
    zorder=100,
    s=100,
)  # colors[2]?
ax.legend()

if save_outputs:
    save_ax_as_pdf(
        ax, save_path=OUTPUT_DIR / f"{sampling_date.date()}/covariance/line_cov_cv.pdf"
    )

# %%
matrix, index = reorder_matrix(
    cov.covariance_,
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
lim = residuals.cov().abs().values.max()
ax = matrix_heatmap(
    matrix,
    title="Graphical Lasso Estimate of VAR Residual Covariance Matrix",
    vmin=-lim,
    vmax=lim,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/covariance/heatmap_cov_matrix.pdf",
    )

# %%
matrix, index = reorder_matrix(
    cov.precision_,
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    infer_limits=True,
    title="Graphical Lasso Estimate of VAR Residual Precision Matrix",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/covariance/heatmap_precision_matrix.pdf",
    )

# %%
matrix, index = reorder_matrix(
    prec_to_pcorr(cov.precision_),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Graphical Lasso Estimate of VAR Residual Partial Correlation Matrix",
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/covariance/heatmap_partial_corr_matrix.pdf",
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
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Generalized Impulse Matrix",
    infer_limits=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/heatmap_generalized_impulse_matrix.pdf",
    )

# %%
matrix, index = reorder_matrix(
    sp.linalg.sqrtm(cov.covariance_),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Matrix Square Root Impulse Matrix",
    infer_limits=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/heatmap_sqrtm_impulse_matrix.pdf",
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
        ["ff_sector_ticker", "mean_mcap"],
        ["ff_sector_ticker", "ticker"],
    )
    primary_labels = index.get_level_values(1).tolist()
    secondary_labels = index.get_level_values(0).tolist()
    ax = matrix_heatmap(
        matrix,
        title=f"VMA({h}) Matrix",
        infer_limits=True,  # , vmin=-lim, vmax=lim,
        labels=primary_labels,
        secondary_labels=secondary_labels,
    )
    if save_outputs:
        save_ax_as_pdf(
            ax,
            save_path=OUTPUT_DIR / f"{sampling_date.date()}/vma/heatmap_vma{h}_matrix.pdf",
        )
    plt.show()

# %% [markdown]
# #### Impulse Response Matrices

# %%
for h in [0, 1, 2, 3, 5, 10, 21]:
    matrix, index = reorder_matrix(
        fevd.impulse_response_functions(h),
        df_info,
        ["ff_sector_ticker", "mean_mcap"],
        ["ff_sector_ticker", "ticker"],
    )
    primary_labels = index.get_level_values(1).tolist()
    secondary_labels = index.get_level_values(0).tolist()
    ax = matrix_heatmap(
        matrix,
        title=f"Impulse Response({h}) Matrix",
        infer_limits=True,
        labels=primary_labels,
        secondary_labels=secondary_labels,
    )
    if save_outputs:
        save_ax_as_pdf(
            ax,
            save_path=OUTPUT_DIR / f"{sampling_date.date()}/irf/heatmap_ir{h}_matrix.pdf",
        )
    plt.show()

# %% [markdown]
# #### Innovation Response Variance Matrices

# %%
for h in [0, 1, 2, 3, 5, 10, 21]:
    matrix, index = reorder_matrix(
        fevd.innovation_response_variances(h),
        df_info,
        ["ff_sector_ticker", "mean_mcap"],
        ["ff_sector_ticker", "ticker"],
    )
    primary_labels = index.get_level_values(1).tolist()
    secondary_labels = index.get_level_values(0).tolist()
    ax = matrix_heatmap(
        matrix,
        title=f"Innovation Response Variance ({h}) Matrix",
        infer_limits=True,
        labels=primary_labels,
        secondary_labels=secondary_labels,
    )
    if save_outputs:
        save_ax_as_pdf(
            ax,
            save_path=OUTPUT_DIR / f"{sampling_date.date()}/irv/heatmap_irv{h}_matrix.pdf",
        )
    plt.show()

# %% [markdown]
# ### Network structure
# #### FEVD

# %%
network = fevd.to_network(table_name="fevd", horizon=21)

# %%
matrix, index = reorder_matrix(
    network.adjacency_matrix - np.diag(np.diag(network.adjacency_matrix)),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="FEVD Adjacency Matrix (off-diagonal values only)",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/heatmap_FEVD_adjacency.pdf",
    )

# %%
pd.Series(index=df_info.ticker.values, data = network.net_connectedness().squeeze()).sort_values()

# %%
_ = draw_network(
    network,
    df_info,
    title=f"FEVD Network ({sampling_date.date()})",
    save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/network_FEVD.png"
    if save_outputs
    else None,
)

# %% [markdown]
# ### Weighted FEVD

# %%
weights = df_info["mean_mcap"].values / df_info["mean_mcap"].values.mean()
network = fevd.to_network(table_name="fevd", horizon=21, weights=weights)

# %%
matrix, index = reorder_matrix(
    network.adjacency_matrix - np.diag(np.diag(network.adjacency_matrix)),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Weighted FEVD Adjacency Matrix (off-diagonal values only)",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/heatmap_WFEVD_adjacency.pdf",
    )

# %%
_ = draw_network(
    network,
    df_info,
    title=f"Weighted FEVD Network ({sampling_date.date()})",
    save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/network_WFEVD.png"
    if save_outputs
    else None,
)

# %%
pd.Series(index=df_info.ticker, data=network.net_connectedness().squeeze()).sort_values().tail(20)

# %% [markdown]
# #### FEV

# %%
network = fevd.to_network(table_name="fev", horizon=21)

# %%
matrix, index = reorder_matrix(
    network.adjacency_matrix - np.diag(np.diag(network.adjacency_matrix)),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="FEV Adjacency Matrix (off-diagonal values only)",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/heatmap_FEV_adjacency.pdf",
    )

# %%
_ = draw_network(
    network,
    df_info,
    title=f"FEV Network ({sampling_date.date()})",
    save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/network_FEV.png"
    if save_outputs
    else None,
)

# %% [markdown]
# ### Weighted FEV

# %%
weights = df_info["mean_mcap"].values / df_info["mean_mcap"].values.mean()
network = fevd.to_network(table_name="fev", horizon=21, weights=weights)

# %%
matrix, index = reorder_matrix(
    network.adjacency_matrix - np.diag(np.diag(network.adjacency_matrix)),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="Weighted FEVD Adjacency Matrix (off-diagonal values only)",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
if save_outputs:
    save_ax_as_pdf(
        ax,
        save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/heatmap_WFEVD_adjacency.pdf",
    )

# %%
_ = draw_network(
    network,
    df_info,
    title=f"Weighted FEV Network ({sampling_date.date()})",
    save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/network_WFEV.png"
    if save_outputs
    else None,
)

# %% [markdown]
# #### Gamma

# %%
from euraculus.network.network import Network

# %%
D = fevd.to_network(table_name="fevd", horizon=21).adjacency_matrix
D_minus = D - np.diag(np.diag(D))
Gamma = np.linalg.inv(np.eye(100) - D_minus)
network = Network(Gamma)

# %%
matrix, index = reorder_matrix(
    network.adjacency_matrix - np.diag(np.diag(network.adjacency_matrix)),
    df_info,
    ["ff_sector_ticker", "mean_mcap"],
    ["ff_sector_ticker", "ticker"],
)
primary_labels = index.get_level_values(1).tolist()
secondary_labels = index.get_level_values(0).tolist()
ax = matrix_heatmap(
    matrix,
    title="FEV Adjacency Matrix (off-diagonal values only)",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    labels=primary_labels,
    secondary_labels=secondary_labels,
)
# if save_outputs:
#     save_ax_as_pdf(
#         ax,
#         save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/heatmap_FEV_adjacency.pdf",
#     )

# %%
_ = draw_network(
    network,
    df_info,
    title="Gamma Network",
    # save_path=OUTPUT_DIR / f"{sampling_date.date()}/network/network_FEV.png"
    # if save_outputs
    # else None,
)

# %%

# %%
pd.Series(index=df_info.ticker.values, data=network.net_connectedness().squeeze()).sort_values()

# %%
# ticker lookup
data.lookup_ticker(
    tickers=["DHR", "PM", "CHTR", "SNOW", "ZM", "SQ", "MU"],
    date=sampling_date,
)

# %%
# ticker lookup
data.lookup_ticker(
    tickers=["PCG"],
    date=dt.datetime(year=1987, month=1, day=31),
)

# %%
# degree distribution
fig, ax = plt.subplots(1, 1)
ax.hist(network.adjacency_matrix.flatten(), bins=100)
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()

# %% [markdown]
# ### Contributions

# %%
network = fevd.to_network(
    table_name="fevd", horizon=21, weights=df_info["mean_mcap"].values
)

# %%
# network = fevd.to_network(table_name = "fev", horizon=21)

# %%
ax = contribution_bars(
    scores=df_info["mean_mcap"].values,
    names=df_info["ticker"],
    title="Market capitalization",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=np.exp(df_log_vola).mean().values,
    names=df_info["ticker"],
    title="Volatility",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.out_connectedness().flatten(),
    names=df_info["ticker"],
    title="Out connectedness",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.amplification_factor().flatten(),
    names=df_info["ticker"],
    title="Amplification Factor",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.absorption_rate().flatten(),
    names=df_info["ticker"],
    title="Absorption Rate",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.in_connectedness().flatten(),
    names=df_info["ticker"],
    title="In-Connectedness",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.out_connectedness().flatten(),
    names=df_info["ticker"],
    title="Out-Connectedness",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.in_eigenvector_centrality().flatten(),
    names=df_info["ticker"],
    title="In-Eigenvector Centrality",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.out_eigenvector_centrality().flatten(),
    names=df_info["ticker"],
    title="Out-Eigenvector Centrality",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.in_page_rank(
        weights=df_info["mean_mcap_volatility"].values.reshape(-1, 1),
        alpha=0.85,
    ).flatten(),
    names=df_info["ticker"],
    title="In-Page Rank (α=0.85)",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.out_page_rank(
        weights=df_info["mean_mcap_volatility"].values.reshape(-1, 1),
        alpha=0.85,
    ).flatten(),
    names=df_info["ticker"],
    title="Out-Page Rank (α=0.85)",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.in_page_rank(
        weights=df_info["mean_mcap_volatility"].values.reshape(-1, 1),
        alpha=0.95,
    ).flatten(),
    names=df_info["ticker"],
    title="In-Page Rank (α=0.95)",
    normalize=False,
)

# %%
ax = contribution_bars(
    scores=network.out_page_rank(
        weights=df_info["mean_mcap_volatility"].values.reshape(-1, 1),
        alpha=0.95,
    ).flatten(),
    names=df_info["ticker"],
    title="Out-Page Rank (α=0.95)",
    normalize=False,
)
