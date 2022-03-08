# -*- coding: utf-8 -*-
# %% [markdown]
# # Regularized FEVD estimation - Single window
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt

import kungfu as kf
import matplotlib.pyplot as plt
import missingno as mno
import networkx as nx
import numpy as np

# %%
import pandas as pd
import scipy as sp
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from euraculus.data import DataMap
from euraculus.var import VAR
from euraculus.covar import AdaptiveThresholdEstimator, GLASSO
from euraculus.fevd import FEVD
from euraculus.utils import make_ticker_dict, autocorrcoef, prec_to_pcorr
from euraculus.plot import (
    corr_heatmap,
    missing_data,
    histogram,
    var_timeseries,
    net_cv_contour,
    net_scatter_losses,
    network_graph,
    cov_cv_contour,
    cov_scatter_losses,
    plot_glasso_cv,
)

# %% [markdown]
# ## Set up
# ### Sampling date

# %%
sampling_date = dt.datetime(year=2019, month=12, day=31)

# %%
sampling_date = dt.datetime(year=2021, month=12, day=31)

# %% [markdown]
# ### Data

# %%
option = "logvar_capm_resid"  # "spy_capm_decomp"

# %%
data = DataMap("../data")
df_idio_var = data.load_historic(sampling_date=sampling_date, column="var_idio")
df_logvar_resid = data.load_historic(
    sampling_date=sampling_date, column="logvar_capm_resid"
)
df_var = data.load_historic(sampling_date=sampling_date, column="var")
df_spy_var = data.load_spy_data(series="var").loc[df_idio_var.index]
df_info = data.load_asset_estimates(
    sampling_date=sampling_date, columns=["ticker", "comnam", "last_size", "mean_size"]
)

# %% [markdown]
# ### Tickers

# %%
ticker_list = (
    data.load_historic(sampling_date=sampling_date, column="ticker")
    .tail(1)
    .values.ravel()
    .tolist()
)
column_to_ticker = make_ticker_dict(ticker_list)

# %% [markdown]
# Make and export table:

# %%
# get data
df_lookup = df_info[["ticker", "comnam"]]

# create output
df_tickers = pd.Series(
    dtype="object", name="Company Name", index=pd.Index(data=[], name="Ticker")
)

# fill in values from lookup
for ticker, unique_ticker in zip(ticker_list, column_to_ticker.values()):
    df_tickers[unique_ticker] = (
        df_lookup[df_lookup.ticker == ticker]
        .comnam.values[0]
        .replace("&", "\&")
        .title()
    )

# store
df_tickers = df_tickers.to_frame().sort_index()
# df_tickers.export_to_latex(filename='tickers.tex', path='../reports/tables/')
data.store(
    data=df_tickers, path="samples/{:%Y-%m-%d}/tickers.csv".format(sampling_date)
)

# %% [markdown]
# ## Data Summary & Processing

# %%
corr_heatmap(
    df_var.corr(),
    title="Total Variance Correlations",
    save_path="../reports/figures/estimation/heatmap_total_variance_correlation.pdf",
)

# %%
corr_heatmap(
    euraculus.utils.autocorrcoef(df_var, lag=1),
    title="Total Variance Auto-Correlations (First order)",
    save_path="../reports/figures/estimation/heatmap_total_variance_autocorrelation.pdf",
)

# %%
missing_data(
    df_idio_var,
    save_path="../reports/figures/estimation/matrix_missing.pdf",
)

# %%
print(
    "{:.2f}% of observations missing".format(
        (df_idio_var.isna()).sum().sum() / len(df_idio_var.stack(dropna=False)) * 100
    )
)

# %%
histogram(
    df_idio_var.fillna(0).stack(),
    title="Distribution of Raw Data",
    drop_tails=0.01,
    bins=100,
    save_path="../reports/figures/estimation/histogram_raw_data.pdf",
)

# %%
var_timeseries(
    df_idio_var,
    total_var=df_var,
    index_var=df_spy_var,
    save_path="../reports/figures/estimation/variance_decomposition.pdf",
)

# %%
if option == "spy_capm_decomp":
    df_log_idio_var = data.log_replace(df_idio_var, method="min")
elif option == "logvar_capm_resid":
    df_log_idio_var = df_logvar_resid

# %%
corr_heatmap(
    df_log_idio_var.corr(),
    title="Idiosyncratic Variance Correlations",
    save_path="../reports/figures/estimation/heatmap_idiosyncratic_variance_correlation.pdf",
)

# %%
corr_heatmap(
    autocorrcoef(df_log_idio_var, lag=1),
    title="Idiosyncratic Variance Auto-Correlations (First order)",
    save_path="../reports/figures/estimation/heatmap_idiosyncratic_variance_autocorrelation.pdf",
)

# %%
pd.Series(np.diag(autocorrcoef(df_log_idio_var, lag=1))).plot(
    kind="hist", title="Diagonal Autocorrelations", bins=20
)
plt.show()

# %%
histogram(
    df_log_idio_var.stack(),
    title="Distribution of Idiosyncratic Variances",
    save_path="../reports/figures/estimation/histogram_idiosyncratic_variance.pdf",
)

# %% [markdown]
# ## VAR estimation

# %% [markdown]
# $r_{i,t} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} f_{k,t} + \sum_{j=1}^{N} \gamma_{i,j} r_{j,t-1}  + u_{i,t}$
#
# where
# - $r_{i,t}$: $1 \times 1$ (asset excess returns)
# - $\alpha_i$: $1 \times 1$ (intercepts/pricing errors)
# - $f_{k,t}$: $1 \times 1$ (factor excess returns)
# - $\beta_{i,k}$: $1 \times 1$ (factor loadings)
# - $\gamma_{i,j}$: $1 \times 1$ (VAR coefficients)
# - $u_{i,t}$: $1 \times 1$ (error term)
#
# Adaptive Elastic Net with hyperparameters $\lambda, \kappa$
#
# $(\hat{\alpha}, \hat{\beta}, \hat{\gamma})= \underset{(\alpha, \beta, \gamma)}{argmin} \Bigg[ \frac{1}{2NT}\sum_{i=1}^{N}\sum_{t=1}^{T} \Big(r_{i,t} - \big(\alpha_i + \sum_{k=1}^{K} \beta_{i,k} f_{k,t} + \sum_{j=1}^{N} \gamma_{i,j} r_{j,t-1}\big)\Big)^2 + \lambda \sum_{i=1}^{N}\sum_{j=1}^{N} w_{i,j} \big(\kappa |\gamma_{i,j}| + (1-\kappa) \frac{1}{2} \gamma_{i,j}^2\big)\Bigg]$
#
# weights are set to $w_{i,j} =|\hat{\beta}_{i,j,OLS}|^{-1}$

# %%
var = VAR(has_intercepts=True, p_lags=1)
var.fit(var_data=df_log_idio_var, method="OLS")

# %%
corr_heatmap(
    var.var_1_matrix_,
    title="Non-regularized VAR(1) coefficients (OLS)",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ols_var1_matrix.pdf",
)

# %%
# for logged data
var_grid = {
    "alpha": np.geomspace(1e-5, 1e0, 11),
    "lambdau": np.geomspace(1e-2, 1e2, 13),
    #'gamma': np.geomspace(1e-2, 1e2, 15),
}

# %%
# %%time
var_cv = var.fit_adaptive_elastic_net_cv(var_data=df_log_idio_var, grid=var_grid, return_cv=True, penalize_diagonals=True)

# %%
net_cv_contour(
    var_cv,
    15,
    logx=True,
    logy=True,
    save_path="../reports/figures/estimation/contour_var.pdf",
)

# %%
net_scatter_losses(
    var_cv,
    save_path="../reports/figures/estimation/scatter_var.pdf",
)

# %%
gammas = var.var_1_matrix_
n_series = df_log_idio_var.shape[1]
density = var.var_density_  # (gammas!=0).sum()/n_series**2*n_series
κ = var_cv.best_params_["alpha"]
λ = var_cv.best_params_["lambdau"]
print("VAR(1) matrix is {:.2f}% dense.".format(density * 100))
print("Best hyperparameters are alpha={:.4f}, lambda={:.4f}.".format(κ, λ))
print(
    "Average VAR spillover is {:.4f}, absolute {:.4f}".format(
        gammas.mean(), abs(gammas).mean()
    )
)
try:
    print(
        "Mean factor loading is {:.4f}, with min {:.4f}, max {:.4f}".format(
            var.exog_loadings_.mean(),
            var.exog_loadings_.min(),
            var.exog_loadings_.max(),
        )
    )
except:
    pass

# %%
corr_heatmap(
    var.var_1_matrix_,
    title="Regularized VAR(1) coefficients (Adaptive Elastic Net)",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_aenet_var1_matrix.pdf",
)

# %%
vargraph = nx.convert_matrix.from_numpy_array(
    var.var_1_matrix_, create_using=nx.DiGraph
)
network_graph(
    vargraph,
    column_to_ticker,
    linewidth=0.02,
    red_percent=2,
    title="VAR network links",
    save_path="../reports/figures/estimation/network_VAR.png",
)

# %%
residuals = var.residuals(df_log_idio_var)

# %%
corr_heatmap(
    residuals.corr(),
    title="VAR Residual Correlation",
    save_path="../reports/figures/estimation/heatmap_VAR_residual_correlation.pdf",
)

# %%
corr_heatmap(
    autocorrcoef(residuals, lag=1),
    title="VAR Residual Auto-Correlation (First order)",
    save_path="../reports/figures/estimation/heatmap_VAR_residual_autocorrelation.pdf",
)

# %% [markdown]
# ### Covariance matrix estimation

# %%
histogram(
    residuals.stack(),
    title="Distribution of VAR Residuals",
    save_path="../reports/figures/estimation/histogram_VAR_residuals.pdf",
)

# %%
corr_heatmap(
    residuals.cov(),
    title="Sample Estimate of the VAR Residual Covariance Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_VAR_residual_covariance.pdf",
)

# %%
corr_heatmap(
    prec_to_pcorr(np.linalg.inv(residuals.cov())),
    title="Sample Estimate of the VAR Residual Partial Correlation Matrix",
    save_path="../reports/figures/estimation/heatmap_VAR_residual_partial_corr.pdf",
)

# %% [markdown]
# #### Adaptive Threshold Estimation

# %%
ate = AdaptiveThresholdEstimator()

# %%
cov_grid = {"delta": np.geomspace(0.5, 1, 11), "eta": np.linspace(0, 2, 13)}

# %%
# %%time
cov_cv = GridSearchCV(
    ate, cov_grid, cv=12, n_jobs=-1, verbose=1, return_train_score=True
).fit(residuals)
covar_ate = cov_cv.best_estimator_.covariance_

# %%
cov_cv_contour(
    cov_cv,
    15,
    logx=False,
    logy=False,
    save_path="../reports/figures/estimation/contour_cov.pdf",
)

# %%
cov_scatter_losses(
    cov_cv,
    save_path="../reports/figures/estimation/scatter_cov.pdf",
)

# %%
density = (covar_ate != 0).sum() / covar_ate.size * 100
δ = cov_cv.best_params_["delta"]
η = cov_cv.best_params_["eta"]
print("covariance matrix is {:.2f}% dense.".format(density))
print("Best hyperparameters are delta={:.2f}, eta={:.10f}.".format(δ, η))
print(
    "Values were shrunk by {:.2f}% on average.".format(
        (1 - (covar_ate / np.cov(residuals, rowvar=False)).mean()) * 100
    )
)

# %%
lim = residuals.cov().abs().values.max()
corr_heatmap(
    covar_ate,
    title="Adaptive Threshold Estimate of VAR Residual Covariance Matrix",
    vmin=-lim,
    vmax=lim,
    save_path="../reports/figures/estimation/heatmap_cov_matrix_ate.pdf",
)

# %%
corr_heatmap(
    residuals.cov().values - covar_ate, "Shrinkage difference", infer_limits=True
)

# %%
corr_heatmap(
    euraculus.utils.prec_to_pcorr(np.linalg.inv(covar_ate)),
    title="Adaptive Threshold Estimate of the VAR Residual Partial Correlation Matrix",
)

# %% [markdown]
# #### GLASSO

# %%
glasso = GLASSO()
glasso_grid = {"alpha": np.geomspace(1e-2, 1e0, 25)}

# %%
# %%time
glasso_cv = GridSearchCV(
    glasso, glasso_grid, cv=12, n_jobs=-1, verbose=1, return_train_score=True
).fit(residuals)
covar = glasso_cv.best_estimator_.covariance_

# %%
plot_glasso_cv(
    glasso_cv,
    save_path="../reports/figures/estimation/line_cov_cv.pdf",
)

# %%
lim = residuals.cov().abs().values.max()
corr_heatmap(
    covar,
    title="Graphical Lasso Estimate of VAR Residual Covariance Matrix",
    vmin=-lim,
    vmax=lim,
    save_path="../reports/figures/estimation/heatmap_cov_matrix.pdf",
)

# %%
corr_heatmap(residuals.cov().values - covar, "Shrinkage difference", infer_limits=True)

# %%
corr_heatmap(
    glasso_cv.best_estimator_.precision_,
    infer_limits=True,
    title="Graphical Lasso Estimate of VAR Residual Precision Matrix",
)

# %%
corr_heatmap(
    euraculus.utils.prec_to_pcorr(glasso_cv.best_estimator_.precision_),
    title="Graphical Lasso Estimate of VAR Residual Partial Correlation Matrix",
    save_path="../reports/figures/estimation/heatmap_partial_corr_matrix.pdf",
)

# %%
glasso_cv.best_estimator_.precision_density_

# %%
covgraph = nx.convert_matrix.from_numpy_array(
    glasso_cv.best_estimator_.covariance_, create_using=nx.Graph
)
network_graph(
    covgraph,
    column_to_ticker,
    linewidth=0.5,
    red_percent=0,
    title="Covariance Matrix as Undirected Graph",
)

# %%
congraph = nx.convert_matrix.from_numpy_array(
    -glasso_cv.best_estimator_.precision_
    + 2 * np.diag(np.diag(glasso_cv.best_estimator_.precision_)),
    create_using=nx.Graph,
)
network_graph(
    congraph,
    column_to_ticker,
    linewidth=1,
    red_percent=0,
    title="Contemporaneous Network Links",
    save_path="../reports/figures/estimation/network_glasso.png",
)

# %% [markdown]
# ### FEVD

# %%
horizon = 21
fevd = FEVD(var.var_1_matrix_, covar)

# %%
lim = abs(fevd.vma_matrix(1)).max()

# %% [markdown]
# #### VMA Matrices

# %%
corr_heatmap(
    fevd.vma_matrix(0),
    title="VMA(0) Matrix",
    infer_limits=True,  # , vmin=-lim, vmax=lim,
    save_path="../reports/figures/estimation/heatmap_vma0_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.vma_matrix(1),
    title="VMA(1) Matrix",
    infer_limits=True,  # , vmin=-lim, vmax=lim,
    save_path="../reports/figures/estimation/heatmap_vma1_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.vma_matrix(2),
    title="VMA(2) Matrix",
    infer_limits=True,  # , vmin=-lim, vmax=lim,
    save_path="../reports/figures/estimation/heatmap_vma2_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.vma_matrix(3),
    title="VMA(3) Matrix",
    infer_limits=True,  # , vmin=-lim, vmax=lim,
    save_path="../reports/figures/estimation/heatmap_vma3_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.vma_matrix(5),
    title="VMA(5) Matrix",
    infer_limits=True,  # , vmin=-lim, vmax=lim,
    save_path="../reports/figures/estimation/heatmap_vma5_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.vma_matrix(10),
    title="VMA(10) Matrix",
    infer_limits=True,  # , vmin=-lim, vmax=lim,
    save_path="../reports/figures/estimation/heatmap_vma10_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.vma_matrix(21),
    title="VMA(21) Matrix",
    infer_limits=True,  # , vmin=-lim, vmax=lim,
    save_path="../reports/figures/estimation/heatmap_vma21_matrix.pdf",
)

# %% [markdown]
# #### Impulse Response Matrices

# %%
corr_heatmap(
    fevd.impulse_response(0),
    title="Impulse Response(0) Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ir0_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.impulse_response(1),
    title="Impulse Response(1) Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ir1_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.impulse_response(2),
    title="Impulse Response(2) Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ir2_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.impulse_response(3),
    title="Impulse Response(3) Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ir3_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.impulse_response(5),
    title="Impulse Response(5) Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ir5_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.impulse_response(10),
    title="Impulse Response(10) Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ir10_matrix.pdf",
)

# %%
corr_heatmap(
    fevd.impulse_response(21),
    title="Impulse Response(21) Matrix",
    infer_limits=True,
    save_path="../reports/figures/estimation/heatmap_ir21_matrix.pdf",
)

# %% [markdown]
# #### FEV Adjacency

# %%
corr_heatmap(
    pd.DataFrame(fevd.fev_single(horizon)) - np.diag(np.diag(fevd.fev_single(horizon))),
    title="FEV Single Contributions",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    save_path="../reports/figures/estimation/heatmap_FEV_contributions.pdf",
)

# %%
corr_heatmap(
    pd.DataFrame(fevd.decompose_fev(horizon=horizon, normalise=False))
    - np.diag(np.diag(fevd.decompose_fev(horizon=horizon, normalise=False))),
    title="FEV Decomposition",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path="../reports/figures/estimation/heatmap_FEV_decomposition.pdf",
)

# %%
corr_heatmap(
    pd.DataFrame(fevd.decompose_fev(horizon=horizon, normalise=True))
    - np.diag(np.diag(fevd.decompose_fev(horizon=horizon, normalise=True))),
    title="FEV Decomposition (row-normalised)",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path="../reports/figures/estimation/heatmap_FEV_decomposition_normalised.pdf",
)

# %% [markdown]
# #### FU Adjacency

# %%
corr_heatmap(
    pd.DataFrame(fevd.fu_single(horizon)) - np.diag(np.diag(fevd.fu_single(horizon))),
    title="FU Single Contributions",
    vmin=0,
    cmap="binary",
    infer_vmax=True,
    save_path="../reports/figures/estimation/heatmap_FU_contributions.pdf",
)

# %%
corr_heatmap(
    pd.DataFrame(fevd.decompose_fu(horizon=horizon, normalise=False))
    - np.diag(np.diag(fevd.decompose_fu(horizon=horizon, normalise=False))),
    title="FU Decomposition",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path="../reports/figures/estimation/heatmap_FU_decomposition.pdf",
)

# %%
corr_heatmap(
    pd.DataFrame(fevd.decompose_fu(horizon=horizon, normalise=True))
    - np.diag(np.diag(fevd.decompose_fu(horizon=horizon, normalise=True))),
    title="FU Decomposition (row-normalised)",
    vmin=0,
    vmax=None,
    cmap="binary",
    save_path="../reports/figures/estimation/heatmap_FU_decomposition_normalised.pdf",
)

# %% [markdown]
# ### Network structure

# %% [markdown]
# #### FEV Adjacency

# %%
network_graph(
    fevd.to_fev_graph(horizon, normalise=False),
    column_to_ticker,
    title="FEVD Network (FEV absolute)",
    red_percent=5,
    linewidth=0.25,
    save_path="../reports/figures/estimation/network_FEV_absolute.png",
)

# %%
network_graph(
    fevd.to_fev_graph(horizon, normalise=True),
    column_to_ticker,
    title="FEVD Network (FEV %)",
    red_percent=2,
    linewidth=0.25,
    save_path="../reports/figures/estimation/network_FEV_normalised.png",
)

# %%
_ = data.lookup_ticker(tickers=["FIS"], date=sampling_date)
_

# %% [markdown]
# #### IRV Adjacency

# %%
mean_size = df_info.mean_size

# %%
irv = fevd.innovation_response_variance(horizon=horizon)
index_decomp = fevd.index_variance_decomposition(
    horizon=horizon, weights=mean_size / mean_size.sum()
)

# %%
corr_heatmap(
    fevd.fev_single(horizon=21), title="FEV", infer_limits=True
)  # weights=mean_size))

# %%
corr_heatmap(
    irv, title="Innovation response variance", infer_limits=True
)  # weights=mean_size))

# %%
fig, ax = plt.subplots(1, 1)
mean_size = df_info.mean_size
ax.bar(height=index_decomp, x=range(100))
# ax.bar(height=mean_size/mean_size.sum(), x=range(100))

# %%
fig, ax = plt.subplots(1, 1)
mean_size = df_info.mean_size
ax.bar(height=mean_size, x=range(100))
# ax.bar(height=mean_size/mean_size.sum(), x=range(100))

# %%
graph = nx.convert_matrix.from_numpy_array(
    np.diag(mean_size) ** 2 @ irv, create_using=nx.DiGraph
)
network_graph(
    graph,
    column_to_ticker,
    title="IRV Network (IRV absolute)",
    red_percent=5,
    linewidth=0.25,
    #                        save_path='../reports/figures/estimation/network_FU_absolute.png',
)

# %%
graph = nx.convert_matrix.from_numpy_array(irv, create_using=nx.DiGraph)
network_graph(
    graph,
    column_to_ticker,
    title="IRV Network (IRV absolute)",
    red_percent=5,
    linewidth=0.25,
    #                        save_path='../reports/figures/estimation/network_FU_absolute.png',
)

# %%
graph = nx.convert_matrix.from_numpy_array(
    irv / irv.sum(axis=1), create_using=nx.DiGraph
)
network_graph(
    graph,
    column_to_ticker,
    title="IRV Network (IRV %)",
    red_percent=5,
    linewidth=0.25,
    #                        save_path='../reports/figures/estimation/network_FU_absolute.png',
)

# %% [markdown]
# #### FU Adjacency

# %%
network_graph(
    fevd.to_fu_graph(horizon, normalise=False),
    column_to_ticker,
    title="FEVD Network (FU absolute)",
    red_percent=5,
    linewidth=0.25,
    save_path="../reports/figures/estimation/network_FU_absolute.png",
)

# %%
network_graph(
    fevd.to_fu_graph(horizon, normalise=True),
    column_to_ticker,
    title="FEVD Network (FU %)",
    red_percent=2,
    linewidth=0.25,
    save_path="../reports/figures/estimation/network_FU_normalised.png",
)

# %%
data.lookup_ticker(tickers=["UNH", "XON"], date=sampling_date)


# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## OLD

# %%
def make_ticker_table(ticker_list):
    ticker_table = kf.FinancialSeries(
        dtype="object", name="Company Name", index=pd.Index(data=[], name="Ticker")
    )
    for ticker in ticker_list:
        if ticker in ["BRK.A", "BRK.B"]:
            data = src.utils.lookup_ticker("BRK", year)
        else:
            data = src.utils.lookup_ticker(ticker, year)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[-1, :]
        ticker_table[ticker] = data["comnam"].replace("&", "\&").title()
    ticker_table = ticker_table.to_frame().sort_index()
    return ticker_table


# %%
ticker_table = make_ticker_table(ticker_list)
ticker_table.export_to_latex(filename="tickers.tex", path="../reports/tables/")

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## TRIALS: Correlation Elastic Net

# %%
from sklearn.linear_model import LinearRegression
from src.net import ElasticNet

_ = src.covar.CorrelationNet()
_.fit_elastic_net(df_volas, return_model=False, lambdau=5e-5, alpha=0)
# _.fit_OLS(df_volas, return_model=False)
src.plot.corr_heatmap(_.coef_.reshape(100, -1))

# %%
src.plot.corr_heatmap(df_volas.corr())

# %%


# %%


# %%


# %%


# %%


# %%
A = fevd.decompose_pct(10)

# %%
B = (A + A.T) / 2

# %%
src.utils.matrix_asymmetry(A)

# %%
src.plot.corr_heatmap(
    abs(A / B - 1),
    vmin=0,
    vmax=1,
    title="Directed links as %  of total link",
    cmap="Greys",
)

# %%


# %%


# %%
from sklearn.decomposition import FactorAnalysis

factors = pd.DataFrame(FactorAnalysis(n_components=1).fit_transform(df_est))
favar = src.FAVAR(df_est, factors, p_lags=1)

# %%
pd.Series(fevd.in_connectedness(horizon=horizon).ravel()).hist()
plt.show()

# %%
pd.Series(fevd.out_connectedness(horizon=horizon).ravel()).hist()
plt.show()

# %%
fevd_10 = fevd.summarize(10)

# %%
create_results_data(df_est, df_ana, favar, cv, fevd, horizon).corr().round(4)

# %% [markdown]
# ## Analysis

# %%
def create_results_data(df_est, df_ana, favar, cv, fevd, horizon):  # graph
    """Creates a dataframe containing estimation results."""
    df = pd.DataFrame(index=favar.var_data.columns)
    df["mean_est_ret"] = df_est.mean() * 252
    df["mean_ana_ret"] = df_ana.mean() * 252
    df["beta"] = favar.factor_loadings_
    df["alpha"] = favar.intercepts_
    df["in_connectedness"] = fevd.in_connectedness(horizon=horizon)
    df["out_connectedness"] = fevd.out_connectedness(horizon=horizon)
    df["fev_others"] = fevd.fev_others(horizon=horizon)
    df["fev_all"] = fevd.fev_all(horizon=horizon)
    df["average_connectedness"] = fevd.average_connectedness(horizon=horizon)
    # df['eigenvector_centrality'] = np.linalg.eig(favar.residual_cov_('LW'))[1][:,0]
    # df['network_centraility'] = nx.eigenvector_centrality(graph)

    return df


# %%
def estimate_single_year(year, selected_factors, grid, horizon):

    # load data
    df_est, df_ana, df_factors_est, df_factors_ana = src.loader.load_year_all(year)
    fac_est, fac_ana = src.loader.select_factor_data(
        df_factors_est, df_factors_ana, selected_factors=selected_factors
    )

    # favar
    favar = src.FAVAR(df_est, fac_est, p_lags=1)
    cv = favar.fit_elastic_net_cv(grid=hyperparameter_grid, return_cv=True)

    # fevd
    fevd = src.FEVD(favar.var_1_matrix_, favar.residual_cov_("LW"))

    # save results
    df_year = create_results_data(df_est, df_ana, favar, cv, fevd, horizon)

    return df_year


# %%
def estimate_single_year_vola(year, selected_factors, grid, horizon):

    # load data
    df_est, df_ana, df_factors_est, df_factors_ana = src.loader.load_year_all_vola(year)
    fac_est, fac_ana = src.loader.select_factor_data(
        df_factors_est, df_factors_ana, selected_factors=selected_factors
    )

    # favar
    favar = src.FAVAR(df_est, fac_est, p_lags=1)
    cv = favar.fit_elastic_net_cv(grid=hyperparameter_grid, return_cv=True)

    # fevd
    fevd = src.FEVD(favar.var_1_matrix_, favar.residual_cov_("LW"))

    # save results
    df_year = create_results_data(df_est, df_ana, favar, cv, fevd, horizon)

    return df_year


# %%
df_results = pd.DataFrame([])
for year in range(2000, 2019):
    df_year = estimate_single_year(
        year=year, selected_factors=["mktrf"], grid=hyperparameter_grid, horizon=10
    )
    df_year.index = pd.MultiIndex.from_product([[year], df_year.index])

    df_results = df_results.append(df_year)
    df_results.to_csv("../data/interim/df_results.csv")
    print("finished year {}".format(year))

# %%
kf.FinancialDataFrame(df_results).fit_linear_regression(
    "mean_est_ret", ["beta", "in_connectedness", "fev_others"]
).summary()

# %%
kf.FinancialDataFrame(df_results).fit_linear_regression(
    "mean_ana_ret", ["beta", "alpha", "in_connectedness", "fev_others"]
).summary()

# %%
kf.FinancialDataFrame(df_results).fit_linear_regression(
    "beta", ["alpha", "in_connectedness", "fev_others"]
).summary()

# %%


# %%
