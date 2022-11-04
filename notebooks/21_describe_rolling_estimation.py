# %% [markdown]
# # Description of Rolling Estimation
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from euraculus.data import DataMap
from euraculus.plot import save_ax_as_pdf
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    TIME_STEP,
)
from kungfu.plotting import add_recession_bars

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import pandas as pd
cmap = mpl.cm.get_cmap('hsv')

# %%
save_outputs = True

# %% [markdown]
# ## Load & prepare data

# %%
# %%time
data = DataMap()
try:
    df_stats = data.read("analysis/df_stats.pkl")
    df_estimates = data.read("analysis/df_estimates.pkl")
    df_summary = data.read("analysis/df_summary.pkl")
except ValueError:
    df_stats = data.load_estimation_summary()
    df_estimates = data.load_asset_estimates()
    df_summary = data.load_selection_summary()
    data.dump(df_stats, "analysis/df_stats.pkl")
    data.dump(df_estimates, "analysis/df_estimates.pkl")
    data.dump(df_summary, "analysis/df_summary.pkl")

# %%
# Additional calculations
df_stats["mean_shrinkage"] = (
    df_stats["var_nonzero_shrinkage"] + df_stats["precision_nonzero_shrinkage"]
) / 2
df_stats["cov_used_df"] = (
    df_stats["precision_density"] * df_stats["N"] ** 2 - df_stats["N"]
) / 2 + df_stats["N"]
df_stats["var_regular_lost_df"] = df_stats["N"] ** 2 + df_stats["N"] + df_stats["N"]
df_stats["covar_regular_lost_df"] = (df_stats["N"] * (df_stats["N"] - 1)) / 2
df_stats["var_estimate_share"] = df_stats["var_regular_lost_df"] / (
    df_stats["var_regular_lost_df"] + df_stats["covar_regular_lost_df"]
)
df_stats["mean_density"] = (
    df_stats["var_matrix_density"] * df_stats["var_estimate_share"]
    + (1 - df_stats["var_estimate_share"]) * df_stats["precision_density"]
)

# %% [markdown]
# ### Plot Sampling Fractions

# %%
# calculations
df_sampling = pd.DataFrame()
df_sampling["n_assets"] = df_summary.groupby("sampling_date").size()
df_sampling["total_mean_mcap"] = df_summary.groupby("sampling_date")[
    "mean_mcap"
].sum()
df_sampling["100_mean_mcap"] = df_summary.groupby("sampling_date").apply(
    lambda x: x.sort_values("mean_valuation_volatility", ascending=False)[
        "mean_mcap"
    ]
    .iloc[:100]
    .sum()
)
df_sampling["total_mean_vv"] = df_summary.groupby("sampling_date")[
    "mean_valuation_volatility"
].sum()
df_sampling["100_mean_vv"] = df_summary.groupby("sampling_date").apply(
    lambda x: x.sort_values("mean_valuation_volatility", ascending=False)[
        "mean_valuation_volatility"
    ]
    .iloc[:100]
    .sum()
)

# set up plot
fig, ax = plt.subplots(figsize=(16, 6))
ax2 = ax.twinx()
ax.set_title("Proportion of our sample compared to the entire CRSP universe")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# plot lines
l1 = ax.plot(
    df_sampling["100_mean_mcap"] / df_sampling["total_mean_mcap"],
    label=f"Market capitalization, mean={(df_sampling['100_mean_mcap'] / df_sampling['total_mean_mcap'] *100).mean().round(2)}%",
    linestyle="-",
)
l2 = ax.plot(
    df_sampling["100_mean_vv"] / df_sampling["total_mean_vv"],
    label=f"Value volatility, mean={(df_sampling['100_mean_vv'] / df_sampling['total_mean_vv'] *100).mean().round(2)}%",
    linestyle="--",
)
l3 = ax2.plot(
    100 / df_sampling["n_assets"],
    label=f"Number of assets (right axis), mean={(100 / df_sampling['n_assets'] * 100).mean().round(2)}%",
    linestyle="-.",
    color=colors[2],
)

# format
ax.set_ylim([0, 1])
ax.set_xlim([df_sampling.index[0], df_sampling.index[-1]])
ax.set_yticks([i / 10 for i in range(11)])
ax.set_yticklabels([f"{int(tick*100)}%" for tick in ax.get_yticks()])
ax.set_ylabel(
    "Market capitalization & Value volatility share"
)  # , color=colors[0])
ax.tick_params(axis="y")  # , labelcolor=colors[0])

ax2.set_ylim([0, 0.1])
ax2.set_xlim([df_sampling.index[0], df_sampling.index[-1]])
ax2.set_yticks([i / 100 for i in range(11)])
ax2.set_yticklabels([f"{int(tick*10)}%" for tick in ax.get_yticks()])
ax2.grid(False)
ax2.set_ylabel("Share of total number of assets", color=colors[2])
ax2.tick_params(axis="y", labelcolor=colors[2])

add_recession_bars(
    ax, freq="M", startdate=df_sampling.index[0], enddate=df_sampling.index[-1]
)

# legend
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels)  # , loc="center left")

if save_outputs:
    save_ax_as_pdf(ax, save_path=f"../reports/rolling/sampling_summary.pdf")

# %%
print(
    max(df_sampling["100_mean_mcap"] / df_sampling["total_mean_mcap"]),
    min(df_sampling["100_mean_mcap"] / df_sampling["total_mean_mcap"]),
)
print(
    max(df_sampling["100_mean_vv"] / df_sampling["total_mean_vv"]),
    min(df_sampling["100_mean_vv"] / df_sampling["total_mean_vv"]),
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax2 = ax.twinx()
ax.set_xlim([-1, 101])
cmap = mpl.cm.get_cmap('nipy_spectral')

for i, year in enumerate(reversed(range(1990, 2022))):
    color=cmap(i/32)

    # prepare data
    date = dt.datetime(year=year, month=12, day=31)
    df_year = df_summary.loc[
        df_summary.index.get_level_values("sampling_date") == date, "last_mcap"
    ]
    df_year = df_year.sort_values(ascending=False).reset_index(drop=True)*1e3

    ax2.plot(pd.Series([0]).append(df_year.iloc[:101].cumsum()) / df_year.sum(),
             # zorder=2,
             color=color,
             alpha=0.8
            )
    ax.scatter(
        x=df_year.iloc[:100].index + 1,
        y=df_year.iloc[:100],
        label=date.year,
        marker="o",
        zorder=1,
        color=color,
    )

# layout
ax.set_title("Firm Size Distribution (100 largest)")
ax.set_ylabel("Market capitalization")
ax.set_xlabel("Size Rank")
ax.set_yscale("log")
ax.grid(True, which="both", linestyle="-")
ax2.grid(False)#, color="gray")
ax2.set_ylim([0, 0.6])
ax2.set_yticklabels([f"{int(tick*100)}%" for tick in ax2.get_yticks()])
ax2.set_ylabel("Cumulative share")
ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

plt.show()

# %%
fig, ax = plt.subplots(1, 1)

# plot scatters
cmap = mpl.cm.get_cmap('nipy_spectral')

for i, year in enumerate(reversed(range(1990, 2022))):
    color=cmap(i/32)

    # prepare data
    date = dt.datetime(year=year, month=12, day=31)
    df_year = df_estimates.loc[
        df_estimates.index.get_level_values("sampling_date") == date
    ]
    # df_year = df_year.sort_values(ascending=False).reset_index(drop=True)*1e3
    
    ax.scatter(
        x=df_year["mean_mcap"]*1e3,
        y=df_year["fev_out_page_rank_85"],
        label=date.year,
        marker="o",
        zorder=1,
        color=color,
    )

# layout
ax.set_title("Size vs. Page rank centrality (logarithmic scale)")
ax.set_ylabel("Centrality")
ax.set_xlabel("Market Capitalization")
ax.grid(True, which="both", linestyle="-")
# ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()

plt.show()

# %% [markdown]
# ## Aggregate plots
# ### Estimation summary

# %%
# set up plot
fig, axes = plt.subplots(3, 1, figsize=(20, 12))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# hyperparameters
ax1 = axes[0]
ax2 = ax1.twinx()
ax1.set_title("Cross-validated hyperparameters")
l1 = ax1.plot(
    df_stats["lambda"],
    linestyle="-",
    label="λ, mean={}".format(df_stats["lambda"].mean().round(2)),
    c=colors[0],
)
l2 = ax1.plot(
    df_stats["rho"],
    linestyle="-.",
    label="ρ, mean={}".format(df_stats["rho"].mean().round(2)),
    c=colors[1],
)
l3 = ax2.plot(
    df_stats["kappa"],
    linestyle="--",
    label="κ, mean={:.1e}".format(df_stats["kappa"].mean()),
    c=colors[2],
)

# l4 = ax2.plot(df['eta'], linestyle=':', label='η, mean={}'.format(df['eta'].mean().round(2)), c=colors[3])
ax1.set_ylim([1e-3, 1e1])
ax2.set_ylim([1e-5, 1e0])
ax1.set_yscale("log")
ax2.set_yscale("log")
ax2.grid(None)
ax1.set_ylabel("Penalty hyperparameters (λ, ρ)", color=colors[0])
ax1.tick_params(axis="y", labelcolor=colors[0])
ax2.set_ylabel("L1 hyperparameter (κ)", color=colors[2])
ax2.tick_params(axis="y", labelcolor=colors[2])
lines = l1 + l2 + l3  # +l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, bbox_to_anchor=(1.05, 0.5), loc="center left")
add_recession_bars(ax1, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1])

# Losses
ax1 = axes[1]
ax2 = ax1.twinx()
ax1.set_title("Cross-validation losses")
ax1.set_ylim([0, 1])
l11 = ax1.plot(
    df_stats["var_cv_loss"],
    linestyle="-",
    label="VAR CV loss, mean={}".format(df_stats["var_cv_loss"].mean().round(2)),
    c=colors[0],
)
l12 = ax1.plot(
    df_stats["var_train_loss"],
    linestyle="--",
    label="VAR train loss, mean={}".format(df_stats["var_train_loss"].mean().round(2)),
    c=colors[0],
)
ax1.set_ylabel("VAR MSE", color=colors[0])
ax1.tick_params(axis="y", labelcolor=colors[0])

# ax2.set_ylim([0, 500])
ax2.grid(None)
l21 = ax2.plot(
    df_stats["covar_cv_loss"],
    linestyle="-.",
    label="Covariance CV loss, mean={}".format(df_stats["covar_cv_loss"].mean().round(2)),
    c=colors[1],
)
l22 = ax2.plot(
    df_stats["covar_train_loss"],
    linestyle=":",
    label="Covariance train loss, mean={}".format(
        df_stats["covar_train_loss"].mean().round(2)
    ),
    c=colors[1],
)
ax2.set_ylabel("Covariance loss", color=colors[1])
ax2.tick_params(axis="y", labelcolor=colors[1])

lines = l11 + l12 + l21 + l22
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, bbox_to_anchor=(1.05, 0.5), loc="center left")
add_recession_bars(ax1, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1])

# R2
ax1 = axes[2]
ax2 = ax1.twinx()
ax1.set_title("Goodness of fit")
ax1.set_ylim([0, 1])
l11 = ax1.plot(
    df_stats["var_r2"],
    label="AEnet, mean={}".format(df_stats["var_r2"].mean().round(2)),
    c=colors[0],
    linestyle="-",
)
l12 = ax1.plot(
    df_stats["var_r2_ols"],
    label="OLS, mean={}".format(df_stats["var_r2_ols"].mean().round(2)),
    c=colors[0],
    linestyle="--",
)
ax1.set_ylabel("VAR R²", color=colors[0])
ax1.tick_params(axis="y", labelcolor=colors[0])

ax2.grid(None)
l21 = ax2.plot(
    df_stats["cov_mean_likelihood"],
    label="GLASSO, mean={}".format(df_stats["cov_mean_likelihood"].mean().round(2)),
    c=colors[1],
    linestyle="-.",
)
l22 = ax2.plot(
    df_stats["cov_mean_likelihood_sample_estimate"],
    label="Sample covariance, mean={}".format(
        df_stats["cov_mean_likelihood_sample_estimate"].mean().round(2)
    ),
    c=colors[1],
    linestyle=":",
)
ax2.set_ylabel("Covariance average log-likelihood", color=colors[1])
ax2.tick_params(axis="y", labelcolor=colors[1])

lines = l11 + l12 + l21 + l22
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, bbox_to_anchor=(1.05, 0.5), loc="center left")
add_recession_bars(ax1, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1])

if save_outputs:
    save_ax_as_pdf(ax, save_path="../reports/rolling/estimation_summary.pdf")

# %% [markdown]
# ### Partial R$^2$

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Network stats
ax.set_title("VAR Partial $R^2$")
ax.plot(
    df_stats["var_partial_r2_factors"],
    label="Partial $R^2$ factors, mean="
    + str((df_stats["var_partial_r2_factors"]).mean().round(2)),
    c=colors[0],
)
ax.plot(
    df_stats["var_partial_r2_var"],
    label="Partial $R^2$ spillovers, mean="
    + str((df_stats["var_partial_r2_var"]).mean().round(2)),
    c=colors[1],
    linestyle="--",
)
ax.set_ylabel("Partial $R^2$")
ax.legend()
add_recession_bars(ax, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1])

if save_outputs:
    save_ax_as_pdf(ax, save_path="../reports/rolling/partial_r2_summary.pdf")

# %% [markdown]
# ### Regularization summary

# %%
# def plot_regularisation_summary(df, save_path=None):
# set up plot
fig, axes = plt.subplots(3, 1, figsize=(20, 12))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
df = df_stats.astype(float)

# Degrees of Freedom
ax = axes[0]
ax.set_title("Degrees of freedom")
ax.fill_between(
    df.index,
    0,
    df["var_df_used"],
    alpha=0.5,
    label="DFs used by VAR estimation, mean={}".format(
        int(df["var_df_used"].mean())
    ),
    color=colors[0],
)
ax.plot(df["var_df_used"], c=colors[0], linewidth=1)
ax.fill_between(
    df.index,
    df["var_df_used"],
    df["var_df_used"] + df["cov_used_df"],
    alpha=0.5,
    label="DFs used by covariance estimation, mean={}".format(
        int(df["cov_used_df"].mean())
    ),
    color=colors[1],
)
ax.plot(df["var_df_used"] + df["cov_used_df"], c=colors[1], linewidth=1)
ax.fill_between(
    df.index,
    df["var_df_used"] + df["cov_used_df"],
    df["nobs"],
    alpha=0.3,
    label="Remaining DFs, mean={}".format(
        int((df["nobs"] - df["var_df_used"] - df["cov_used_df"]).mean())
    ),
    color=colors[2],
)
ax.plot(
    df["nobs"],
    c=colors[2],
    label="Total data points, mean={}".format(int(df["nobs"].mean())),
)
ax.plot(
    df["var_regular_lost_df"],
    c=colors[0],
    label="Non-regularised VAR DFs ({})".format(
        int(df["var_regular_lost_df"].mean())
    ),
    linestyle="--",
    linewidth=1.5,
)
ax.plot(
    df["var_regular_lost_df"] + df["covar_regular_lost_df"],
    c=colors[1],
    label="Non-regularised total DFs ({})".format(
        int((df["var_regular_lost_df"] + df["covar_regular_lost_df"]).mean())
    ),
    linestyle="-.",
    linewidth=1.5,
)
ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

# Sparsity
ax = axes[1]
ax.set_title("Estimate sparsity")
ax.plot(
    1 - df["var_matrix_density"],
    linestyle="-",
    label="VAR matrix sparsity, mean={}".format(
        (1 - df["var_matrix_density"]).mean().round(2)
    ),
)
ax.plot(
    1 - df["precision_density"],
    linestyle="--",
    label="Precision matrix sparsity, mean={}".format(
        (1 - df["precision_density"]).mean().round(2)
    ),
)
ax.plot(
    1 - df["mean_density"],
    linestyle="-.",
    label="Overall estimate sparsity, mean={}".format(
        (1 - df["mean_density"]).mean().round(2)
    ),
)
ax.set_ylim([0, 1])
ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

# Shrinkage
ax = axes[2]
ax.set_title("Estimate shrinkage")
ax.plot(
    df["var_nonzero_shrinkage"],
    linestyle="-",
    label="VAR matrix shrinkage, mean={}".format(
        (df["var_nonzero_shrinkage"]).mean().round(2)
    ),
)
ax.plot(
    df["precision_nonzero_shrinkage"],
    linestyle="--",
    label="Precision matrix shrinkage, mean={}".format(
        (df["precision_nonzero_shrinkage"]).mean().round(2)
    ),
)
# ax.plot(
#     df["mean_shrinkage"],
#     linestyle=":",
#     label="Overall estimate shrinkage, mean={}".format(
#         (df["mean_shrinkage"]).mean().round(2)
#     ),
# )
ax.plot(
    df["covar_nonzero_shrinkage"],
    linestyle="-.",
    label="Covariance matrix shrinkage, mean={}".format(
        (df["covar_nonzero_shrinkage"]).mean().round(2)
    ),
)
ax.set_ylim([0, 1])
ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")
add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

if save_outputs:
    save_ax_as_pdf(ax, save_path="../reports/rolling/regularisation_summary.pdf")

# %% [markdown]
# ### Network summary

# %%
# set up plot
fig, axes = plt.subplots(1, 1, figsize=(20, 8))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# connectedness
ax1 = axes  # [0]
l1 = ax1.plot(
    df_stats["fevd_avg_connectedness"],
    label="Average connectedness $c^{avg}$, mean="
    + str((df_stats["fevd_avg_connectedness"]).mean().round(2)),
    c=colors[0],
)
ax1.set_ylabel("Connectedness", color=colors[0])
ax1.tick_params(axis="y", labelcolor=colors[0])
ax1.set_ylim([0, 0.6])
# ax1.set_yscale("log")

# concentration
ax2 = ax1.twinx()
l2 = ax2.plot(
    df_stats["fevd_concentration_out_connectedness_herfindahl"].rolling(1).mean(),
    label="Network concentration, mean={}".format(
        (df_stats["fevd_concentration_out_connectedness_herfindahl"]).mean().round(2)
    ),
    linestyle="--",
    c=colors[1],
)
ax2.grid(None)
ax2.set_ylabel("Concentration", color=colors[1])
ax2.tick_params(axis="y", labelcolor=colors[1])
ax2.set_ylim([0, 0.06])
# ax3.set_yscale("log")

# asymmetry
ax3 = ax1.twinx()
l3 = ax3.plot(
    df_stats["fevd_asymmetry"],
    label="Network directedness, mean={}".format(
        (df_stats["fevd_asymmetry"]).mean().round(2)
    ),
    linestyle="-.",
    c=colors[2],
)
ax3.grid(None)
ax3.set_ylabel("Directedness", color=colors[2])
ax3.tick_params(axis="y", labelcolor=colors[2])
ax3.yaxis.set_label_coords(1.07, 0.5)
ax3.tick_params(direction="out", pad=50)
ax3.set_ylim([0, 0.6])
# ax2.set_yscale("log")

# # amplification
# ax4 = ax1.twinx()
# l4 = ax4.plot(
#     df_stats["fevd_amplification"],
#     label="Network amplification, mean={}".format(
#         (df_stats["fevd_amplification"]).mean().round(2)
#     ),
#     linestyle=(0, (5, 1)),
#     c=colors[3],
# )
# ax4.grid(None)
# ax4.set_ylabel("Amplification", color=colors[3])
# ax4.tick_params(axis="y", labelcolor=colors[3])
# ax4.yaxis.set_label_coords(1.115, 0.5)
# ax4.tick_params(direction="out", pad=100)
# ax4.set_ylim([0.2, 1.4])
# # ax2.set_yscale("log")

# figure formatting
ax1.set_title("FEVD Network statistics")
# lines = l1 + l3 + l4

lines = l1 + l2 + l3  # + l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")
add_recession_bars(ax1, freq="M", startdate=df.index[0], enddate=df.index[-1])

if save_outputs:
    save_ax_as_pdf(ax, save_path="../reports/rolling/network_summary.pdf")

# %% [markdown]
# ### Ledoit-Wolf test for diagonal generalized innovations

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

#
ax1 = ax
l1 = ax1.plot(
    df_stats["innovation_diagonality_test_stat"],
    label="test statistic, mean="
    + str((df_stats["innovation_diagonality_test_stat"]).mean().round(2)),
    c=colors[0],
)
ax1.set_ylabel("test statistic", color=colors[0])
ax1.tick_params(axis="y", labelcolor=colors[0])

#
ax2 = ax1.twinx()
l2 = ax2.plot(
    df_stats["innovation_diagonality_p_value"].astype(float),
    label="p-value, mean="
    + str((df_stats["innovation_diagonality_test_stat"]).mean().round(2)),
    c=colors[1],
    linestyle="--",
)
ax2.grid(False)
ax2.set_ylim([-0.01, 1.01])
ax2.set_ylabel("p-value", color=colors[1])
ax2.tick_params(axis="y", labelcolor=colors[1])

# legend
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center left")
add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

if save_outputs:
    save_ax_as_pdf(ax, save_path="../reports/rolling/ledoitwolf_test.pdf")

# %% [markdown]
# ### Average factor loading

# %%
df_estimates["var_factor_loadings_crsp"].unstack().mean(axis=1).plot()

# %% [markdown]
# ## Granular analysis

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
df_estimates["fev_fullout_connectedness"] = df_estimates["fev_out_connectedness"] + df_estimates["fev_self_connectedness"]
df_estimates["fev_fullout_connectedness_weighted"] = df_estimates["fev_out_connectedness_weighted"] + df_estimates["fev_self_connectedness_weighted"]
df_estimates["fevd_fullout_connectedness"] = df_estimates["fevd_out_connectedness"] + df_estimates["fevd_self_connectedness"]
df_estimates["fevd_fullout_connectedness_weighted"] = df_estimates["fevd_out_connectedness_weighted"] + df_estimates["fevd_self_connectedness_weighted"]
df_estimates["fev_net_connectedness"] = df_estimates["fev_out_connectedness"] - df_estimates["fev_in_connectedness"]
df_estimates["fevd_net_connectedness"] = df_estimates["fevd_out_connectedness"] - df_estimates["fevd_in_connectedness"]

# %%
df_estimates.iloc[-100:, :][["fev_out_page_rank_85", "ticker", "mean_mcap_rank", "ff_sector_ticker"]].sort_values("fev_out_page_rank_85", ascending=False).iloc[:10]

# %%
df_estimates[["fev_out_page_rank_85", "ticker", "comnam", "sic_division", "ff_sector", "ff_sector_ticker", "gics_sector"]].groupby("sampling_date").apply(lambda x: x.sort_values("fev_out_page_rank_85", ascending=False))

# %%
# influence_variable = "fevd_fullout_connectedness"
influence_variable = "fevd_out_connectedness_weighted"
# influence_variable = "fev_fullout_connectedness"
# influence_variable = "fev_out_eigenvector_centrality"
# influence_variable = "fev_out_page_rank_85"
# nfluence_variable = "mean_mcap"

_df = df_estimates[[influence_variable, "ticker", "comnam", "sic_division", "ff_sector", "ff_sector_ticker", "gics_sector"]]#.reset_index().set_index(["sampling_date", "ticker"])
_df[[ "sic_division", "ff_sector", "ff_sector_ticker", "gics_sector"]] = _df[[ "sic_division", "ff_sector", "ff_sector_ticker", "gics_sector"]].fillna("N/A")
influence = _df[influence_variable].unstack()
tickers = _df["ticker"].unstack()
comnams = _df["comnam"].unstack()
sics = _df["sic_division"].unstack()
ffs = _df["ff_sector"].unstack()
gics = _df["gics_sector"].unstack()
rank = influence.rank(axis=1, ascending=False)

# %%
# %%time
df_influencers = pd.DataFrame(index=tickers.index)

for i in range(1, 101):
    i_tickers = tickers[rank==i].astype(str).min(axis=1).rename(i)
    i_comnams = comnams[rank==i].astype(str).min(axis=1).rename(i)
    i_sics = sics[rank==i].astype(str).min(axis=1).rename(i)
    i_ffs = ffs[rank==i].astype(str).min(axis=1).rename(i)
    i_gics = gics[rank==i].astype(str).min(axis=1).rename(i)
    
    
    df_influencers = df_influencers.join(i_tickers + ": " + i_comnams)
    
df_influencers.to_csv("df_influencers.csv")

# %%
fig, ax = plt.subplots(figsize=(24,10))

ax.stackplot(influence.index,
             (influence).fillna(0).iloc[:, :].values.T,
             # labels=influence.columns,
             alpha=1.)
ax.set_xlim([influence.index[0], influence.index[-1]])


# # add influencer tick label
# ax2 = ax.twiny()
# ax2.grid(False)
# ax.set_xlim([influence.index[0], influence.index[-1]])
# ax2.set_xticks(df_influencers.index)
# ax2.set_xticklabels(df_influencers[1], rotation=90, fontsize=6)

# ax.legend(loc='upper left')
ax.set_title(f"Total value uncertainty decomposition: {influence_variable}")
ax.set_xlabel('Date')
ax.set_ylabel(influence_variable)

plt.show()


# %%
# industry_totals.iloc[:,industry_totals.mean().rank(ascending=False).values-1]

# %%
def construct_weights(df_estimates: pd.DataFrame, weighting_variable: str, grouping_variables: list = None):
    """
    
    Args:
        df_estimates:
        weighting_variable:
        grouping_variables:
    
    Returns:
        df_weights:
    """
    _df = df_estimates.copy()
    if grouping_variables is not None:
        _df = _df.groupby(grouping_variables)[weighting_variable].sum().unstack()
    else:
        _df = _df[weighting_variable].unstack()
        
    df_weights = _df / _df.sum(axis=1).values.reshape(-1, 1)
    
    return df_weights


# %%
mcap_weights = construct_weights(df_estimates=df_estimates, weighting_variable="mean_mcap", grouping_variables=["sampling_date", "permno"]) 
network_weights = construct_weights(df_estimates=df_estimates, weighting_variable="fevd_fullout_connectedness_weighted", grouping_variables=["sampling_date", "permno"])
df_leverage = network_weights / mcap_weights


# %%
def plot_leverage(df_estimates: pd.DataFrame, grouping_variables: list = None):
    """"""
    mcap_weights = construct_weights(df_estimates=df_estimates, weighting_variable="mean_mcap", grouping_variables=grouping_variables) 
    network_weights = construct_weights(
        df_estimates=df_estimates, 
        weighting_variable="fevd_fullout_connectedness_weighted",
        grouping_variables=grouping_variables,
    )
    df_leverage = network_weights / mcap_weights
    
    
    # set up line colors
    sector_colors = plt.get_cmap("Paired")(np.linspace(0, 1, 12))
    ff_sector_tickers = [
        "NoDur",
        "Durbl",
        "Manuf",
        "Enrgy",
        "Chems",
        "BusEq",
        "Telcm",
        "Utils",
        "Shops",
        "Hlth",
        "Money",
        "Other",
    ]
    ff_sector_codes = {tick: i for i, tick in enumerate(ff_sector_tickers)}
    
    fig, ax = plt.subplots(1, 1)
    count = 0
    for permno, series in df_leverage.iteritems():
        ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ticker"]
        comnam = df_estimates.loc[(series.dropna().index[-1], permno), "comnam"]
        ff_sector_ticker = df_estimates.loc[(series.dropna().index[-1], permno), "ff_sector_ticker"]
        max_leverage = series.max()
        max_leverage_pos = series.argmax()
        
        if max_leverage > 2:
            ax.plot(series, color=sector_colors[ff_sector_codes[ff_sector_ticker]], linewidth=1.5, alpha=1)
            ax.text(series.index[max_leverage_pos], max_leverage+0.05, s=ticker, horizontalalignment="center", fontsize=8)
            ax.scatter(series.index[max_leverage_pos], max_leverage, color=sector_colors[ff_sector_codes[ff_sector_ticker]])
            count += 1#(series > 2).sum()
        else:
            ax.plot(series, color=sector_colors[ff_sector_codes[ff_sector_ticker]], linewidth=0.5, alpha=0.5)
        
    add_recession_bars(ax, startdate=df_leverage.index[0], enddate=df_leverage.index[-1])
    # legends
    sector_legend = ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=sector_colors[i], label=ticker)
            for ticker, i in ff_sector_codes.items()
        ],
        title="Sectors",
        edgecolor="k",
        facecolor="lightgray",
        bbox_to_anchor=(1, 0.5),
        loc="center left",
    )
    ax.set_xlim([df_leverage.index[0], df_leverage.index[-1]])
    ax.set_title("Leverage effect of FEVD spillover network on non-systematic innovations in the wealth portfolio")
    ax.set_ylabel("Wealth effect multiplier")
    print(count)

# %%
plot_leverage(df_estimates)

# %%

# %%
# %%time
df_hist_ret = data.load_nonoverlapping_historic()#columns=["retadj"])

# %%
# %%time
df_fut_ret = data.load_nonoverlapping_future()#columns=["retadj"])


# %%
def make_indices(df_estimates, df_historic, df_future):
    # create weights
    mcap_weights = construct_weights(df_estimates=df_estimates, weighting_variable="mean_mcap", grouping_variables=["sampling_date", "permno"]) 
    network_weights = construct_weights(df_estimates=df_estimates, weighting_variable="fevd_fullout_connectedness_weighted", grouping_variables=["sampling_date", "permno"])
    
    # 
    df_historic = df_historic.join(
        mcap_weights.stack().rename("mcap_weights"), on=["sampling_date", "permno"]
    ).join(
        network_weights.stack().rename("network_weights"), on=["sampling_date", "permno"]
    )
    df_future = df_future.join(
        mcap_weights.stack().rename("mcap_weights"), on=["sampling_date", "permno"]
    ).join(
        network_weights.stack().rename("network_weights"), on=["sampling_date", "permno"]
    )
    
    df_indices = pd.DataFrame()
    df_indices["hist_vw"] = (df_historic["mcap_weights"] * df_historic["retadj"]).unstack().sum(axis=1)
    df_indices["hist_nw"] = (df_historic["network_weights"] * df_historic["retadj"]).unstack().sum(axis=1)
    df_indices["hist_nw_resid"] = ((df_historic["network_weights"] - df_historic["mcap_weights"]) * df_historic["retadj"]).unstack().sum(axis=1)
    df_indices["hist_lw_resid"] = ((df_historic["network_weights"]/df_historic["mcap_weights"])/(df_historic["network_weights"]/df_historic["mcap_weights"]).sum() * df_historic["capm_resid"]).unstack().sum(axis=1)
    
    df_indices["fut_vw"] = (df_future["mcap_weights"] * df_future["retadj"]).unstack().sum(axis=1)
    df_indices["fut_nw"] = (df_future["network_weights"] * df_future["retadj"]).unstack().sum(axis=1)
    df_indices["fut_nw_resid"] = ((df_future["network_weights"] - df_future["mcap_weights"]) * df_future["retadj"]).unstack().sum(axis=1)
    df_indices["fut_lw_resid"] = ((df_future["network_weights"]/df_future["mcap_weights"])/(df_future["network_weights"]/df_future["mcap_weights"]).sum() * df_future["capm_resid"]).unstack().sum(axis=1)
       
    return df_indices


# %%
df_indices = make_indices(df_estimates, df_hist_ret, df_fut_ret)

# %%
df_indices.corr()

# %%
df_spy = data.load_index_estimates()

# %%
df_spy.unstack()[("ret_excess_next1M", "vw")].plot()

# %%
df_indices.resample('M').apply(lambda x: (x+1).prod()-1).join(df_spy.unstack()[[("ret_excess", "vw"),("ret_excess_next12M", "vw")]]).corr()

# %%
df_indices.join(df_spy.unstack()[("ret_excess_next1M", "spy")])

# %%

# %%

# %%
_.columns

# %%
_ = make_index(df_fut_ret)

fig, axes = plt.subplots(3, 1, figsize=(20, 12))

ax = axes[0]
index = (_["mcap_weights"] * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Market capitalization weighted index")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

ax = axes[1]
index = (_["network_weights"] * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Network weighted index")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

ax = axes[2]
index = ((_["network_weights"]/_["mcap_weights"])/(_["network_weights"]/_["mcap_weights"]).sum() * _["capm_resid"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Network residual")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

plt.show()

# %%
_ = make_index(df_hist_ret)

fig, axes = plt.subplots(3, 1, figsize=(20, 12))

ax = axes[0]
index = (_["mcap_weights"] * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Market capitalization weighted index")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

ax = axes[1]
index = (_["network_weights"] * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Network weighted index")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

ax = axes[2]
index = ((_["network_weights"]-_["mcap_weights"]) * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Network residual")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

plt.show()

# %%
_ = make_index(df_fut_ret)

fig, axes = plt.subplots(3, 1, figsize=(20, 12))

ax = axes[0]
index = (_["mcap_weights"] * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Market capitalization weighted index")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

ax = axes[1]
index = (_["network_weights"] * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Network weighted index")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

ax = axes[2]
index = ((_["network_weights"]-_["mcap_weights"]) * _["retadj"]).unstack().sum(axis=1)
ax.plot(index, label="Daily index return")
ax.plot(index.rolling(252).apply(lambda x: (x+1).prod()-1), label="Rolling annualized return")
ax.plot(index.rolling(252).apply(lambda x: x.std() * np.sqrt(252)), label="Rolling annualized volatility")
ax.set_title("Network residual")
add_recession_bars(ax, startdate=index.index[0], enddate=index.index[-1])
ax.set_xlim([index.index[0], index.index[-1]])
ax.legend()

plt.show()

# %%
index = ((_["network_weights"]-_["mcap_weights"]) * _["retadj"]).unstack().sum(axis=1)

# %%
index.rolling(252).apply(lambda x: (x+1).prod()-1).mean()

# %%
index.index[index.rolling(252).apply(lambda x: (x+1).prod()-1).argmin()]

# %%
index.rolling(252).apply(lambda x: (x+1).prod()-1).dropna().sort_values()

# %%

# %%

# %%
df_hist_ret.join(mcap_weights.stack().rename("mcap_weights"), on=["sampling_date", "permno"])

# %%

# %%
autocorr = []
for i, col in network_weights.iteritems():
    if col.notna().sum() > 24:
        autocorr += [col.autocorr()]
        
sum(autocorr)/len(autocorr)

# %%
df_estimates[df_estimates.ticker == "NCR"]

# %%
mcap_weights = df_estimates.groupby(["sampling_date", "ff_sector_ticker"])["mean_mcap"].sum().unstack()
mcap_weights = mcap_weights / mcap_weights.sum(axis=1).values.reshape(-1, 1)

network_weights = df_estimates.groupby(["sampling_date", "ff_sector_ticker"])["fevd_fullout_connectedness_weighted"].sum().unstack()
network_weights = network_weights / network_weights.sum(axis=1).values.reshape(-1, 1)

(network_weights / mcap_weights).plot(cmap="Paired")

# %%

# %%

# %%
from kungfu.plotting import add_recession_bars

# %%
# influence_variable = "fev_out_page_rank_85"
# influence_variable = "fev_out_eigenvector_centrality"
# influence_variable = "fev_out_page_rank_95"
industry_totals = _df.groupby(["sampling_date", "ff_sector_ticker"])[influence_variable].sum().unstack()
industry_totals = industry_totals / industry_totals.sum(axis=1).values.reshape(-1, 1)

fig, ax = plt.subplots(figsize=(22,10))
ax.stackplot(industry_totals.index,
             # (industry_totals.fillna(0).values/(industry_totals.sum(axis=1).to_frame().values)).T,
             (industry_totals).fillna(0).iloc[:, :].values.T,
             labels=industry_totals.columns,
             alpha=1,
             colors=plt.get_cmap("Paired")(np.linspace(0,1,12))
            )
ax.set_xlim([industry_totals.index[0], industry_totals.index[-1]])
ax.set_ylim([0, 1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), title='Sectors', bbox_to_anchor=(1, 0.5), loc="center left")

ax.set_title(f"Page rank decomposition of valuation uncertainty by sector ({influence_variable})")
ax.set_xlabel('Date')
ax.set_ylabel("Contribution share")
ax.set_axisbelow(False)
ax.grid(True, linestyle=":", axis="y")
ax.grid(False, axis="x")

add_recession_bars(ax, freq="M", startdate=industry_totals.index[0], enddate=industry_totals.index[-1])

plt.show()

# %%
# influence_variable = "fev_out_page_rank_85"
# influence_variable = "fev_out_eigenvector_centrality"
# influence_variable = "fev_out_page_rank_95"
industry_totals = _df.groupby(["sampling_date", "ff_sector_ticker"])[influence_variable].sum().unstack()
industry_totals = industry_totals / industry_totals.sum(axis=1).values.reshape(-1, 1)

fig, ax = plt.subplots(figsize=(22,10))
ax.stackplot(industry_totals.index,
             # (industry_totals.fillna(0).values/(industry_totals.sum(axis=1).to_frame().values)).T,
             (industry_totals).fillna(0).iloc[:, :].values.T,
             labels=industry_totals.columns,
             alpha=1,
             colors=plt.get_cmap("Paired")(np.linspace(0,1,12))
            )
ax.set_xlim([industry_totals.index[0], industry_totals.index[-1]])
ax.set_ylim([0, 1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), title='Sectors', bbox_to_anchor=(1, 0.5), loc="center left")

ax.set_title(f"Page rank decomposition of valuation uncertainty by sector ({influence_variable})")
ax.set_xlabel('Date')
ax.set_ylabel("Contribution share")
ax.set_axisbelow(False)
ax.grid(True, linestyle=":", axis="y")
ax.grid(False, axis="x")

add_recession_bars(ax, freq="M", startdate=industry_totals.index[0], enddate=industry_totals.index[-1])

plt.show()

# %% [markdown]
# which shocks are amplified?

# %%

# %%

# %%

# %%
df_estimates[
    ["fev_out_connectedness",
              "fev_out_eigenvector_centrality",
              "fev_out_page_rank",
              # "fev_out_entropy",
              "fev_fullout_connectedness",
              "fev_self_connectedness",
              "mean_size"]
].corr("spearman")


# %%

# %%
def bump(df: pd.DataFrame, column: str, n_tickers: int):
    """"""
    # filter by rank
    df["rank"] = df[column].groupby("sampling_date").rank(ascending=False)
    df = df[df["rank"] <= n_tickers]

    df = df[[column, "rank", "ticker", "comnam"]]
    # df = df[df.index.get_level_values("sampling_date").month == 12]
    
    return df


# %%
column = "fev_out_page_rank"

_ = bump(df_estimates, column=column, n_tickers=100)["rank"].unstack().rolling(12).mean()
_[_>10] = np.nan
_ = _[_.index.get_level_values("sampling_date").month == 12]

_.plot()

# %%
bump(df_estimates, column="fev_out_eigenvector_centrality", n_tickers=100)["rank"].unstack().rolling(12, min_periods=6).mean()#.plot()

# %%
bump(df=df_estimates, column="fev_out_connectedness", n_tickers=1).comnam.values

# %%

# %%

# %%
date = "2000-12-31"
df_estimates.reset_index().set_index(["sampling_date", "ticker"])["fev_out_connectedness"][date].sort_values().plot.bar(title=date)#.pie(title=date)

# %%
date = "2000-12-31"
df_estimates.reset_index().set_index(["sampling_date", "ticker"])["fev_fullout_connectedness"][date].sort_values().plot.bar(title=date)#.pie(title=date)

# %%
date = "2000-12-31"
df_estimates.reset_index().set_index(["sampling_date", "ticker"])["fev_out_eigenvector_centrality"][date].sort_values().plot.bar(title=date)#.pie(title=date)

# %%
date = "2021-12-31"
df_estimates.reset_index().set_index(["sampling_date", "ticker"])["fev_out_page_rank"][date].sort_values().iloc[-10:].plot.pie(title=date, normalize=True)

# %%
df_estimates.reset_index().set_index(["sampling_date", "ticker"])["fev_out_page_rank"].sort_values(ascending=False).head(50)

# %%
df_estimates["fev_out_page_rank"].groupby("sampling_date").apply(lambda x: ((x/x.sum())**2).sum()).plot()

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

# %%
df_stats[[
    # 'fevd_avg_connectedness',
    #     'fevd_avg_connectedness_normalized', 
        'fev_avg_connectedness',
        # 'irv_avg_connectedness',
        # 'fevd_asymmetry',
        # 'fevd_asymmetry_normalized',
        'fev_asymmetry',
       #    'irv_asymmetry',
       #    'fevd_asymmetry_offdiag',
       # 'fevd_asymmetry_normalized_offdiag',
          'fev_asymmetry_offdiag',
       # 'irv_asymmetry_offdiag',
       #    'fevd_concentration_in_connectedness',
       'fev_concentration_in_connectedness',
       # 'irv_concentration_in_connectedness',
       # 'fevd_concentration_out_connectedness',
       'fev_concentration_out_connectedness',
       # 'irv_concentration_out_connectedness',
       # 'fevd_concentration_in_eigenvector_centrality',
       'fev_concentration_in_eigenvector_centrality',
       # 'irv_concentration_in_eigenvector_centrality',
       # 'fevd_concentration_out_eigenvector_centrality',
       'fev_concentration_out_eigenvector_centrality',
       # 'irv_concentration_out_eigenvector_centrality',
       # 'fevd_concentration_in_page_rank',
          'fev_concentration_in_page_rank',
       # 'irv_concentration_in_page_rank',
       #    'fevd_concentration_out_page_rank',
       'fev_concentration_out_page_rank',
          # 'irv_concentration_out_page_rank',
         ]].corr()

# %%
(df_stats['fevd_avg_connectedness']*df_stats['fevd_concentration_out_connectedness']).plot()
