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
# # Rolling Factor FEVD estimation
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

import datetime as dt

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from euraculus.models.covariance import GLASSO, AdaptiveThresholdEstimator
from euraculus.data.map import DataMap
from euraculus.network.fevd import FEVD
from euraculus.models.var import FactorVAR

from euraculus.models.estimate import (
    load_estimation_data,
    estimate_fevd,
)
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
    FACTORS,
    VAR_GRID,
    COV_GRID,
    HORIZON,
    FIRST_ESTIMATION_DATE,
)

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
from euraculus.models.spill import FactorNetworkModel
from euraculus.network.network import Network
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

# %%
data = DataMap(DATA_DIR)

# %%
AGGREGATE_PROXY = "crsp_ew"

# %% [markdown]
# ## Estimation
# ### Test single period

# %%
sampling_date = FIRST_ESTIMATION_DATE #dt.datetime(year=2022, month=3, day=31)

# %%
# %%time
# load data
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
factor_data = data.load_historic_aggregates(sampling_date, column=AGGREGATE_PROXY).to_frame()
return_data = data.load_historic(sampling_date, "retadj").fillna(0)
fevd = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")

# estimate
network_proxy = fevd.forecast_error_variance_decomposition(21)
model = FactorNetworkModel(network_proxy=network_proxy)
model.fit(return_data=return_data, factor_data=factor_data, method="MLE")

# %% [markdown]
# ### Rolling Window

# %%
# %%time
sampling_date = FIRST_ESTIMATION_DATE
while sampling_date <= LAST_SAMPLING_DATE:
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
    factor_data = data.load_historic_aggregates(sampling_date, column=AGGREGATE_PROXY).to_frame()
    return_data = data.load_historic(sampling_date, "retadj").fillna(0)
    fevd = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/fevd.pkl")

    # estimate
    network_proxy = fevd.forecast_error_variance_decomposition(21)
    model = FactorNetworkModel(network_proxy=network_proxy)
    model.fit(return_data=return_data, factor_data=factor_data, method="MLE")

    # store estimates
    data.dump(data=model, path=f"samples/{sampling_date:%Y-%m-%d}/return_model.pkl")
    data.dump(data=return_data, path=f"samples/{sampling_date:%Y-%m-%d}/return_data.pkl")
    data.dump(data=factor_data, path=f"samples/{sampling_date:%Y-%m-%d}/return_aggregate_proxy.pkl")

    # increment monthly end of month
    print(f"Completed estimation at {sampling_date:%Y-%m-%d}")
    sampling_date += TIME_STEP

# %%

# %%

# %%
a: float = 2107/2040
b: float = -5/102
em = a + b * 1.09

# %%
# %%time
sampling_date = FIRST_ESTIMATION_DATE
c_list = []
w_list = []
llf_list = []
grp = pd.Series()
while sampling_date <= LAST_SAMPLING_DATE:

    # load estimates & data
    model = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/return_model.pkl")
    return_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/return_data.pkl")
    factor_data = data.read(path=f"samples/{sampling_date:%Y-%m-%d}/return_aggregate_proxy.pkl")
    df_info = data.load_asset_estimates(sampling_date=sampling_date)
    weights = df_info["mean_mcap"].values / df_info["mean_mcap"].values.sum()
    factor_data = data.load_historic_aggregates(sampling_date, column=AGGREGATE_PROXY).to_frame()
    return_data = data.load_historic(sampling_date, "retadj").fillna(0)

    # calculate
    c_list += [model.scaling_coefficient_]
    w_list += [model.wald_tests_[1][-1]]
    llf_list += [model.res_.fun]
    grp = grp.append(
            pd.Series(
                index=pd.MultiIndex.from_product([[sampling_date], df_info.index], names=["sampling_date", "permno"]),
                data=model.granular_risk_premia(return_data=return_data, factor_data=factor_data, weights=weights) *-b/em,
            )
        )

    # store estimates
    # data.dump(data=model, path=f"samples/{sampling_date:%Y-%m-%d}/return_model.pkl")
    # data.dump(data=return_data, path=f"samples/{sampling_date:%Y-%m-%d}/return_data.pkl")
    # data.dump(data=factor_data, path=f"samples/{sampling_date:%Y-%m-%d}/return_aggregate_proxy.pkl")

    # increment monthly end of month
    print(f"Completed calculation at {sampling_date:%Y-%m-%d}")
    sampling_date += TIME_STEP

# %%
s = pd.Series(
    index=pd.MultiIndex.from_product([[sampling_date], df_info.index], names=["sampling_date", "permno"]),
    data=model.granular_risk_premia(return_data=return_data, factor_data=factor_data, weights=weights) *-b/em * 252,
    )

# %%
pd.MultiIndex.from_product

# %%
s.append(s)

# %%

# %%

# %%

# %%
a: float = 2107/2040
b: float = -5/102
em = a + b * 1.09

weights = df_info["mean_mcap"].values / df_info["mean_mcap"].values.sum()
pd.Series(index=df_info["ticker"], data=model.granular_risk_premia(return_data=return_data, factor_data=factor_data, weights=weights) *-b/em * 252).sort_values().plot(kind="bar")

# %%
pd.Series(
    index=df_info["ticker"],
    data=model.expected_returns(return_data=return_data, factor_data=factor_data, weights=weights) *-b/em * 252
    ).sort_values().plot(kind="bar")

# %%
pd.Series(
    index=df_info["ticker"],
    data=model.expected_aggregate_returns(factor_data=factor_data, weights=weights) *-b/em * 252
    ).sort_values().plot(kind="bar")

# %%
pd.Series(
    index=df_info["ticker"],
    data=model.expected_granular_returns(return_data=return_data, factor_data=factor_data, weights=weights) *-b/em * 252
    ).sort_values().plot(kind="bar")

# %%
Network(adjacency_matrix=model.spillover_matrix_).average_connectedness()

# %%

# %%

# %%

# %%

# %%
model.fit(return_data=return_data, factor_data=factor_data, method="MLE")

# %%
from euraculus.models.spill import StructuralFactorModel

# %%
sfm = StructuralFactorModel(structural_proxy=fevd.forecast_error_variance_decomposition(21))
sfm.fit(return_data=return_data, factor_data=factor_data, method="MLE")

# %%
model.log_likelihood_

# %%
sfm.log_likelihood_

# %%
sfm.scaling_coefficient_

# %%
model.adjusted_intercepts_.mean(), model.adjusted_factor_loadings_.mean()

# %%
sfm.intercepts_.mean(), sfm.factor_loadings_.mean()

# %%
m = FactorNetworkModel(network_proxy=fevd.forecast_error_variance_decomposition(21))
# m2 = FactorNetworkModel(fevd.forecast_error_variances(21))
m3 = FactorNetworkModel(network_proxy=np.ones((100, 100))/100)
m4 = FactorNetworkModel(network_proxy=np.random.randn(100, 100))

# %%
return_data = data.load_historic(sampling_date, "retadj")
factor_data = df_factors[["crsp_ew"]]
m.fit(return_data=return_data, factor_data=factor_data, method="MLE")
# m2.fit(return_data=return_data, factor_data=factor_data, method="MLE")
m3.fit(return_data=return_data, factor_data=factor_data, method="MLE")
m4.fit(return_data=return_data, factor_data=factor_data, method="MLE")

# %%
(m.log_likelihood_,
# m2.log_likelihood_,
m3.log_likelihood_,
m4.log_likelihood_,
)

# %%
(m.scaling_coefficient_,
#  m2.scaling_coefficient_,
 m3.scaling_coefficient_,
 m4.scaling_coefficient_,
)

# %%
(m.wald_tests_[0][-1],
#  m2.wald_tests_[0][-1],
 m3.wald_tests_[0][-1],
 m4.wald_tests_[0][-1],
)

# %%
(m.wald_tests_[1][-1],
#  m2.wald_tests_[1][-1],
 m3.wald_tests_[1][-1],
 m4.wald_tests_[1][-1],
)

# %%
m.intercepts_.mean(), m.factor_loadings_.mean(), m.adjusted_factor_loadings_.mean(), m.scaling_coefficient_

# %%
m.r2(return_data=return_data, factor_data=factor_data,)

# %%
m.partial_r2s(return_data=return_data, factor_data=factor_data,)

# %%
m.component_r2s(return_data=return_data, factor_data=factor_data,)

# %%
np.sqrt(np.diag(m.residual_covariance(return_data=return_data, factor_data=factor_data,)) *252)

# %%
a: float = 2107/2040
b: float = -5/102
em = a + b * 1.09
em

# %%
a: float = 2107/2040
b: float = -5/102
em = a + b * 1.09

weights = df_info["mean_mcap"].values / df_info["mean_mcap"].values.mean()
pd.Series(index=df_info["ticker"], data=-m.granular_risk_premia(return_data=return_data, factor_data=factor_data, weights=weights) *-b/em * 252).sort_values().plot(kind="bar")

# %%
(m.granular_risk_premia(return_data=return_data, factor_data=factor_data,)*-b/em * 252).mean()

# %%
m.aggregate_risk_premia(factor_data=factor_data,)# *-b/em * 252

# %%
(1+ m.aggregate_risk_premia(factor_data=factor_data,) *-b/em) ** 252 -1

# %%
m.inv_structural_matrix_.sum(axis=1)

# %%
m.spillover_matrix_.mean() * 99

# %%
Network(adjacency_matrix=m.spillover_matrix_).average_connectedness()
