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
# # Description of Rolling Estimation
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from euraculus.data.map import DataMap
from euraculus.utils.plot import save_ax_as_pdf
from euraculus.settings import (
    OUTPUT_DIR,
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    AMEX_INCLUSION_DATE,
    NASDAQ_INCLUSION_DATE,
    SPLIT_DATE,
    TIME_STEP,
    ESTIMATION_WINDOW,
    COLORS,
    LAST_ANALYSIS_DATE,
)
from kungfu.plotting import add_recession_bars

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import pandas as pd

cmap = mpl.cm.get_cmap("hsv")

# %%
save_outputs = True

# %% [markdown]
# ## Load & prepare data

# %%
# %%time
data = DataMap()
df_summary = data.read("analysis/df_summary.pkl")[:LAST_ANALYSIS_DATE]
df_stats = data.read("analysis/df_stats.pkl")[:LAST_ANALYSIS_DATE]
df_index = data.read("analysis/df_index.pkl")[:LAST_ANALYSIS_DATE]
df_estimates = data.read("analysis/df_estimates.pkl")[:LAST_ANALYSIS_DATE]
df_indices = data.read("analysis/df_daily_indices.pkl")[:LAST_ANALYSIS_DATE]

# %%
df_stats.index.get_loc(SPLIT_DATE) / len(df_stats)

# %%
df_stats["cov_used_df"] = (
    df_stats["precision_density"] * df_stats["N"] ** 2 - df_stats["N"]
) / 2 + df_stats["N"]
# df_stats["sigma_used_df"] = (
#     df_stats["precision_density_ret"] * df_stats["N"] ** 2 - df_stats["N"]
# ) / 2 + df_stats["N"]
df_stats["var_regular_lost_df"] = df_stats["N"] ** 2 + df_stats["N"] + df_stats["N"]
df_stats["covar_regular_lost_df"] = (df_stats["N"] * (df_stats["N"] - 1)) / 2
df_stats["sigma_regular_lost_df"] = (df_stats["N"] * (df_stats["N"] - 1)) / 2

# %%
# Additional calculations
df_stats["mean_shrinkage"] = (
    df_stats["var_nonzero_shrinkage"] + df_stats["precision_nonzero_shrinkage"]
) / 2
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
df_sampling["total_mean_mcap"] = df_summary.groupby("sampling_date")["mean_mcap"].sum()
df_sampling["100_mean_mcap"] = df_summary.groupby("sampling_date").apply(
    lambda x: x.sort_values("mean_mcap_volatility", ascending=False)["mean_mcap"]
    .iloc[:100]
    .sum()
)
df_sampling["total_mean_vv"] = df_summary.groupby("sampling_date")[
    "mean_mcap_volatility"
].sum()
df_sampling["100_mean_vv"] = df_summary.groupby("sampling_date").apply(
    lambda x: x.sort_values("mean_mcap_volatility", ascending=False)[
        "mean_mcap_volatility"
    ]
    .iloc[:100]
    .sum()
)
df_sampling = df_sampling[NASDAQ_INCLUSION_DATE:]

# set up plot
fig, ax = plt.subplots(figsize=(16, 8))
ax2 = ax.twinx()
ax.set_title("Proportion of our sample compared to the entire CRSP universe")

# plot lines
l1 = ax.plot(
    df_sampling["100_mean_mcap"] / df_sampling["total_mean_mcap"],
    label=f"Market capitalization, mean={(df_sampling['100_mean_mcap'] / df_sampling['total_mean_mcap'] *100).mean().round(2)}%",
    linestyle="-",
)
l2 = ax.plot(
    df_sampling["100_mean_vv"] / df_sampling["total_mean_vv"],
    label=f"Market capitalization volatility, mean={(df_sampling['100_mean_vv'] / df_sampling['total_mean_vv'] *100).mean().round(2)}%",
    linestyle="--",
)
l3 = ax2.plot(
    100 / df_sampling["n_assets"],
    label=f"Number of assets (right axis), mean={(100 / df_sampling['n_assets'] * 100).mean().round(2)}%",
    linestyle="-.",
    color=COLORS[2],
)


# # splits
# ax.axvline(AMEX_INCLUSION_DATE, linestyle=":", linewidth=2, color="grey")
# ax.text(s="AMEX", x=AMEX_INCLUSION_DATE, y=0.9)
# ax.axvline(NASDAQ_INCLUSION_DATE, linestyle=":", linewidth=2, color="grey")
# ax.text(s="NASDAQ", x=NASDAQ_INCLUSION_DATE, y=0.9)
ax.axvline(SPLIT_DATE, linestyle=":", linewidth=2, color="grey")
ax.text(s="First estimation", x=SPLIT_DATE, y=0.9)

# format
ax.set_ylim([0, 1])
ax.set_xlim([df_sampling.index[0], df_sampling.index[-1]])
ax.set_yticks([i / 10 for i in range(11)])
ax.set_yticklabels([f"{int(tick*100)}%" for tick in ax.get_yticks()])
ax.set_ylabel("Market capitalization & Volatility share")  # , color=colors[0])
ax.tick_params(axis="y")  # , labelcolor=colors[0])

ax2.set_ylim([0, 0.2])
ax2.set_xlim([df_sampling.index[0], df_sampling.index[-1]])
ax2.set_yticks([i / 50 for i in range(11)])
ax2.set_yticklabels([f"{int(tick*20)}%" for tick in ax.get_yticks()])
ax2.grid(False)
ax2.set_ylabel("Share of total number of assets", color=COLORS[2])
ax2.tick_params(axis="y", labelcolor=COLORS[2])

add_recession_bars(
    ax, freq="M", startdate=df_sampling.index[0], enddate=df_sampling.index[-1]
)

# legend
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels)  # , loc="center left")

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/sampling_summary.pdf")

# %%
(100 / df_sampling["n_assets"][SPLIT_DATE:]).mean()

# %%
(df_sampling["100_mean_mcap"] / df_sampling["total_mean_mcap"])[SPLIT_DATE:].mean()

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
cmap = mpl.cm.get_cmap("nipy_spectral")

for i, year in enumerate(reversed(range(1990, 2022))):
    color = cmap(i / 32)

    # prepare data
    date = dt.datetime(year=year, month=12, day=31)
    df_year = df_summary.loc[
        df_summary.index.get_level_values("sampling_date") == date, "last_mcap"
    ]
    df_year = df_year.sort_values(ascending=False).reset_index(drop=True) * 1e3

    ax2.plot(
        pd.Series([0]).append(df_year.iloc[:101].cumsum()) / df_year.sum(),
        # zorder=2,
        color=color,
        alpha=0.8,
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
ax2.grid(False)  # , color="gray")
ax2.set_ylim([0, 0.6])
ax2.set_yticklabels([f"{int(tick*100)}%" for tick in ax2.get_yticks()])
ax2.set_ylabel("Cumulative share")
ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

plt.show()

# %%
fig, ax = plt.subplots(1, 1)

# plot scatters
cmap = mpl.cm.get_cmap("nipy_spectral")

for i, year in enumerate(reversed(range(1990, 2022))):
    color = cmap(i / 32)

    # prepare data
    date = dt.datetime(year=year, month=12, day=31)
    df_year = df_estimates.loc[
        df_estimates.index.get_level_values("sampling_date") == date
    ]
    # df_year = df_year.sort_values(ascending=False).reset_index(drop=True)*1e3

    ax.scatter(
        x=df_year["mean_mcap"] * 1e3,
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
# ## Indices

# %%
df_plot = pd.DataFrame()
df_plot["value-weighted logvariance index"] = np.exp(df_indices["logvola_vw"])[SPLIT_DATE:]*np.sqrt(252)
df_plot["equally-weighted logvariance index"] = np.exp(df_indices["logvola_ew"])[SPLIT_DATE:]*np.sqrt(252)
df_plot["value-weighted variance index"] = df_indices["vola_vw"][SPLIT_DATE:]*np.sqrt(252)
df_plot["equally-weighted variance index"] = df_indices["vola_ew"][SPLIT_DATE:]*np.sqrt(252)
df_plot["VIX"] = df_indices["vix_Close"]/100
df_plot.plot()

# %%
df_plot.corr()

# %%
df_plot = pd.DataFrame()
df_plot["value-weighted logvariance index"] = np.exp(df_indices["logvola_vw_99"])[SPLIT_DATE:]*np.sqrt(252)
df_plot["equally-weighted logvariance index"] = np.exp(df_indices["logvola_ew_99"])[SPLIT_DATE:]*np.sqrt(252)
df_plot["value-weighted variance index"] = df_indices["vola_vw_99"][SPLIT_DATE:]*np.sqrt(252)
df_plot["equally-weighted variance index"] = df_indices["vola_ew_99"][SPLIT_DATE:]*np.sqrt(252)
df_plot["VIX"] = df_indices["vix_Close"]/100
df_plot.plot()

# %%
df_plot.corr()

# %%
fig, ax = plt.subplots(1,1)
ax.plot(np.exp(df_indices["logvola_vw"])[SPLIT_DATE:]*np.sqrt(252)*100, label="value-weighted logvariance index", linewidth=0.8, linestyle="-")
ax.plot(np.exp(df_indices["logvola_ew"])[SPLIT_DATE:]*np.sqrt(252)*100, label="equally-weighted logvariance index", linewidth=0.8, linestyle="--")
ax.plot((df_indices["vola_vw"])[SPLIT_DATE:]*np.sqrt(252)*100, label="value-weighted variance index", linewidth=0.8, linestyle="-")
ax.plot((df_indices["vola_ew"])[SPLIT_DATE:]*np.sqrt(252)*100, label="equally-weighted variance index", linewidth=0.8, linestyle="--")
ax.plot(df_indices["vix_Close"], label="VIX", linewidth=0.8)# linestyle="-.")
ax.set_xlim([SPLIT_DATE, df_indices.index[-1]])
ax.legend()
# add_recession_bars(ax)
plt.show()

# %%
fig, ax = plt.subplots(1,1)
ax.plot(np.exp(df_indices["logvola_vw"])[SPLIT_DATE:]*np.sqrt(252)*100, label="value-weighted variance index", linewidth=0.8, linestyle="-")
ax.plot(np.exp(df_indices["logvola_ew"])[SPLIT_DATE:]*np.sqrt(252)*100, label="equally-weighted variance index", linewidth=0.8, linestyle="--")
ax.plot(df_indices["vix_Close"], label="VIX", linewidth=0.8, linestyle="-.")
ax.set_xlim([SPLIT_DATE, df_indices.index[-1]])
ax.legend()
add_recession_bars(ax)
plt.show()

# %%
fig, axes = plt.subplots(4, 1 , figsize=(20, 10))

axes[0].plot((df_indices["logvola_ew"]), color=COLORS[0], label="equally-weighted variance index")
add_recession_bars(axes[0])
axes[0].set_xlim([SPLIT_DATE, df_indices.index[-1]])
axes[0].legend()

axes[1].plot((df_indices["logvola_vw"]), color=COLORS[1], label="value-weighted variance index")
add_recession_bars(axes[1])
axes[1].set_xlim([SPLIT_DATE, df_indices.index[-1]])
axes[1].legend()

axes[2].plot(df_indices["vix_Close"], color=COLORS[2], label="VIX")
add_recession_bars(axes[2])
axes[2].set_xlim([SPLIT_DATE, df_indices.index[-1]])
axes[2].legend()


axes[3].plot(np.sqrt(df_indices["ret_ew"]**2), color=COLORS[3], label="Realized volatility of equally-weighted returns")
add_recession_bars(axes[3])
axes[3].set_xlim([SPLIT_DATE, df_indices.index[-1]])
axes[3].legend()

plt.show()

# %%
(df_indices[["logvola_ew", "logvola_vw", "vix_Close"]])[SPLIT_DATE:].corr()

# %% [markdown]
# ## Aggregate plots
# ### Estimation summary

# %%
df_stats = df_stats[SPLIT_DATE:]

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax2 = ax.twinx()
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

# elements
l1 = ax.plot(
    df_stats["lambda"],
    linestyle="-",
    label="λ, mean={}".format(df_stats["lambda"].mean().round(2)),
    c=COLORS[0],
)
l2 = ax.plot(
    df_stats["rho"],
    linestyle="-.",
    label="ρ, mean={}".format(df_stats["rho"].mean().round(2)),
    c=COLORS[1],
)
l3 = ax2.plot(
    df_stats["kappa"],
    linestyle="--",
    label="κ, mean={:.1e} (right axis)".format(df_stats["kappa"].mean()),
    c=COLORS[2],
)
add_recession_bars(
    ax, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1]
)

# formatting
ax.set_ylim([1e-3, 1e1])
ax.set_yscale("log")
ax.set_ylabel("Penalty hyperparameters (λ, ρ)", color=COLORS[0])
ax.tick_params(axis="y", labelcolor=COLORS[1])

ax2.set_ylim([1e-5, 1e0])
ax2.set_yscale("log")
ax2.set_ylabel("L1 hyperparameter (κ)", color=COLORS[2])
ax2.tick_params(axis="y", labelcolor=COLORS[2])
ax2.grid(None)

# legend
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="lower center", ncol=3)#, bbox_to_anchor=(1.05, 0.5)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/hyperparameters.pdf")

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax2 = ax.twinx()
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

# elements
ax.set_ylim([0, 1])
l11 = ax.plot(
    df_stats["var_cv_loss"],
    linestyle="-",
    label="VAR CV loss, mean={}".format(df_stats["var_cv_loss"].mean().round(2)),
    c=COLORS[0],
)
l12 = ax.plot(
    df_stats["var_train_loss"],
    linestyle="--",
    label="VAR train loss, mean={}".format(df_stats["var_train_loss"].mean().round(2)),
    c=COLORS[0],
)
l21 = ax2.plot(
    df_stats["covar_cv_loss"],
    linestyle="-.",
    label="Covariance CV loss, mean={}".format(df_stats["covar_cv_loss"].mean().round(2)),
    c=COLORS[1],
)
l22 = ax2.plot(
    df_stats["covar_train_loss"],
    linestyle=":",
    label="Covariance train loss, mean={}".format(
        df_stats["covar_train_loss"].mean().round(2)
    ),
    c=COLORS[1],
)
add_recession_bars(ax, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1])

# formatting
ax.set_ylim([0, 1])
ax.set_ylabel("VAR MSE", color=COLORS[0])
ax.tick_params(axis="y", labelcolor=COLORS[0])

ax2.set_ylim([0, 0.05])
ax2.set_ylabel("Covariance loss", color=COLORS[1])
ax2.tick_params(axis="y", labelcolor=COLORS[1])
ax2.grid(None)

# legend
lines = l11 + l12 + l21 + l22
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="lower center", ncol=10)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/losses.pdf")

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax2 = ax.twinx()
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

# elements
l11 = ax.plot(
    df_stats["var_r2"],
    label="AEnet, mean={}".format(df_stats["var_r2"].mean().round(2)),
    c=COLORS[0],
    linestyle="-",
)
l12 = ax.plot(
    df_stats["var_r2_ols"],
    label="OLS, mean={}".format(df_stats["var_r2_ols"].mean().round(2)),
    c=COLORS[0],
    linestyle="--",
)
l21 = ax2.plot(
    df_stats["cov_mean_likelihood"],
    label="GLASSO, mean={}".format(df_stats["cov_mean_likelihood"].mean().round(2)),
    c=COLORS[1],
    linestyle="-.",
)
l22 = ax2.plot(
    df_stats["cov_mean_likelihood_sample_estimate"],
    label="Sample covariance, mean={}".format(
        df_stats["cov_mean_likelihood_sample_estimate"].mean().round(2)
    ),
    c=COLORS[1],
    linestyle=":",
)
add_recession_bars(
    ax, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1]
)

# formatting
ax.set_ylim([0, 1])
ax.set_ylabel("VAR R²", color=COLORS[0])
ax.tick_params(axis="y", labelcolor=COLORS[0])

ax2.set_ylim([-50, 10])
ax2.set_ylabel("Covariance average log-likelihood", color=COLORS[1])
ax2.tick_params(axis="y", labelcolor=COLORS[1])
ax2.grid(None)

# legend
lines = l11 + l12 + l21 + l22
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="lower center", ncol=10)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fit.pdf")

# %% [markdown]
# ### Partial R$^2$

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Network stats
# ax.set_title("VAR Partial $R^2$")
ax.plot(
    df_stats["var_partial_r2_factors"],
    label="Partial $R^2$ factors, mean="
    + str((df_stats["var_partial_r2_factors"]).mean().round(2)),
    c=COLORS[0],
)
ax.plot(
    df_stats["var_partial_r2_var"],
    label="Partial $R^2$ spillovers, mean="
    + str((df_stats["var_partial_r2_var"]).mean().round(2)),
    c=COLORS[1],
    linestyle="--",
)
ax.set_ylabel("Partial $R^2$")
ax.legend()
add_recession_bars(
    ax, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1]
)
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/partial_r2_summary.pdf")

# %% [markdown]
# ### Regularization summary

# %%
df = df_stats.astype(float)

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

# elements
# ax.set_title("Degrees of freedom")
ax.fill_between(
    df.index,
    0,
    df["var_df_used"],
    alpha=0.5,
    label="DFs used by VAR estimation, mean={}".format(int(df["var_df_used"].mean())),
    color=colors[0],
)
ax.plot(df["var_df_used"], c=COLORS[0], linewidth=1)
ax.fill_between(
    df.index,
    df["var_df_used"],
    df["var_df_used"] + df["cov_used_df"],
    alpha=0.5,
    label="DFs used by covariance estimation, mean={}".format(
        int(df["cov_used_df"].mean())
    ),
    color=COLORS[1],
)
ax.plot(df["var_df_used"] + df["cov_used_df"], c=COLORS[1], linewidth=1)
ax.fill_between(
    df.index,
    df["var_df_used"] + df["cov_used_df"],
    df["nobs"],
    alpha=0.3,
    label="Remaining DFs, mean={}".format(
        int((df["nobs"] - df["var_df_used"] - df["cov_used_df"]).mean())
    ),
    color=COLORS[2],
)
ax.plot(
    df["nobs"],
    c=COLORS[2],
    label="Total data points, mean={}".format(int(df["nobs"].mean())),
)
ax.plot(
    df["var_regular_lost_df"],
    c=COLORS[0],
    label="Non-regularised VAR DFs ({})".format(int(df["var_regular_lost_df"].mean())),
    linestyle="--",
    linewidth=1.5,
)
ax.plot(
    df["var_regular_lost_df"] + df["covar_regular_lost_df"],
    c=COLORS[1],
    label="Non-regularised total DFs ({})".format(
        int((df["var_regular_lost_df"] + df["covar_regular_lost_df"]).mean())
    ),
    linestyle="-.",
    linewidth=1.5,
)
add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

# formatting
ax.set_ylim([0, None])
ax.legend(loc="lower center", ncol=3)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/degrees_of_freedom.pdf")

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

# elements
# ax.set_title("Estimate sparsity")
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
add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

# formatting
ax.set_ylim([0, 1])
ax.legend(loc="lower center", ncol=3)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/sparsity.pdf")

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

# elements
# ax.set_title("Estimate shrinkage")
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
add_recession_bars(
    ax, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1]
)

# formatting
ax.set_ylim([0, 1])
ax.legend(loc="upper center", ncol=3)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/shrinkage.pdf")

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))
ax.set_xlim([df_stats.index[0], df_stats.index[-1]])

# elements
# ax.plot(df["cov_df_used_ret"], c=COLORS[0], linewidth=1)
# ax.plot(
#     df["nobs_ret"],
#     c=COLORS[2],
#     label="Total data points, mean={}".format(int(df["nobs"].mean())),
# )
# ax.plot(df["sigma_regular_lost_df"],
#     c=COLORS[1],
#     label="Non-regularised total DFs ({})".format(
#         int((df["sigma_regular_lost_df"]).mean())
#     ),
#     linestyle="-.",
#     linewidth=1.5,
# )

ax.plot(
    1 - df["precision_density_ret"],
    linestyle="--",
    label="Precision matrix sparsity, mean={}".format(
        (1 - df["precision_density_ret"]).mean().round(2)
    ),
)
ax.plot(
    df["precision_nonzero_shrinkage_ret"],
    linestyle="--",
    label="Precision matrix shrinkage, mean={}".format(
        (df["precision_nonzero_shrinkage"]).mean().round(2)
    ),
)
# ax.plot(
#     df["covar_nonzero_shrinkage_ret"],
#     linestyle="-.",
#     label="Covariance matrix shrinkage, mean={}".format(
#         (df["covar_nonzero_shrinkage"]).mean().round(2)
#     ),
# )

add_recession_bars(ax, freq="M", startdate=df.index[0], enddate=df.index[-1])

# formatting
ax.set_ylim([0, None])
ax.legend(loc="lower center", ncol=3)

# if save_outputs:
#     save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/degrees_of_freedom.pdf")

# %% [markdown]
# ### Network summary

# %%
df_stats[["fevd_avg_connectedness", "fevd_concentration_out_connectedness_herfindahl"]].corr().iloc[0, 1]

# %%
df_stats["fevd_avg_connectedness"][df_stats["fevd_avg_connectedness"]>0.2]

# %%
fig, ax = plt.subplots(1, 1, figsize=(14, 4))

ax.plot(df_stats["fevd_avg_connectedness"][SPLIT_DATE:], label = "FEVD connectedness", color=COLORS[0])
ax.plot(df_stats["fevd_avg_connectedness"].rolling(12).mean()[SPLIT_DATE:], label = "12-Month moving average", color="k", linestyle=":")
add_recession_bars(
    ax, freq="M", startdate=df_stats[SPLIT_DATE:].index[0], enddate=df_stats[SPLIT_DATE:].index[-1]
)
ax.set_xlim(SPLIT_DATE, df_stats.index[-1])
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fevd_connectedness.pdf")

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

ax.plot(df_stats["fevd_asymmetry"][SPLIT_DATE:], label = "FEVD directedness", color=COLORS[1])
ax.plot(df_stats["fevd_asymmetry"].rolling(12).mean()[SPLIT_DATE:], label = "12-Month moving average", color="k", linestyle=":")
add_recession_bars(
    ax, freq="M", startdate=df_stats[SPLIT_DATE:].index[0], enddate=df_stats[SPLIT_DATE:].index[-1]
)
ax.set_xlim(SPLIT_DATE, df_stats.index[-1])
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fevd_directedness.pdf")

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

ax.plot(df_stats["fevd_concentration_out_connectedness_herfindahl"][SPLIT_DATE:], label = "FEVD concentration", color=COLORS[2])
ax.plot(df_stats["fevd_concentration_out_connectedness_herfindahl"].rolling(12).mean()[SPLIT_DATE:], label = "12-Month moving average", color="k", linestyle=":")
add_recession_bars(
    ax, freq="M", startdate=df_stats[SPLIT_DATE:].index[0], enddate=df_stats[SPLIT_DATE:].index[-1]
)
ax.set_xlim(SPLIT_DATE, df_stats.index[-1])
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/fevd_concentration.pdf")

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

ax.plot(df_stats["wfevd_avg_connectedness"][SPLIT_DATE:], label = "wFEVD connectedness", color=COLORS[0])
ax.plot(df_stats["wfevd_avg_connectedness"].rolling(12).mean()[SPLIT_DATE:], label = "12-Month moving average", color="k", linestyle=":")
add_recession_bars(
    ax, freq="M", startdate=df_stats[SPLIT_DATE:].index[0], enddate=df_stats[SPLIT_DATE:].index[-1]
)
ax.set_xlim(SPLIT_DATE, df_stats.index[-1])
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_connectedness.pdf")

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

ax.plot(df_stats["wfevd_asymmetry"][SPLIT_DATE:], label = "wFEVD directedness", color=COLORS[1])
ax.plot(df_stats["wfevd_asymmetry"].rolling(12).mean()[SPLIT_DATE:], label = "12-Month moving average", color="k", linestyle=":")
add_recession_bars(
    ax, freq="M", startdate=df_stats[SPLIT_DATE:].index[0], enddate=df_stats[SPLIT_DATE:].index[-1]
)
ax.set_xlim(SPLIT_DATE, df_stats.index[-1])
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_directedness.pdf")

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

ax.plot(df_stats["wfevd_concentration_out_connectedness_herfindahl"][SPLIT_DATE:], label = "wFEVD concentration", color=COLORS[2])
ax.plot(df_stats["wfevd_concentration_out_connectedness_herfindahl"].rolling(12).mean()[SPLIT_DATE:], label = "12-Month moving average", color="k", linestyle=":")
add_recession_bars(
    ax, freq="M", startdate=df_stats[SPLIT_DATE:].index[0], enddate=df_stats[SPLIT_DATE:].index[-1]
)
ax.set_xlim(SPLIT_DATE, df_stats.index[-1])
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/wfevd_concentration.pdf")

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

ax.plot(df_index['var_annual'].unstack()[SPLIT_DATE:]["sample_ew"]**0.5, label = "Sample average", color=COLORS[3])
ax.plot(df_index['var_annual'].unstack()[SPLIT_DATE:]["crsp_vw"]**0.5, label = "CRSP value-weighted", color=COLORS[1], linestyle="--")
ax.set_ylim([0, None])
ax.set_xlim(SPLIT_DATE, df_index['var_annual'].unstack().index[-1])
add_recession_bars(
    ax, freq="M", startdate=SPLIT_DATE, enddate=df_index['var_annual'].unstack().index[-1]
)
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/volatility.pdf")

# %%
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

ax.plot(df_index['ret_excess'].unstack()[SPLIT_DATE:]["sample_ew"], label = "Sample average", color=COLORS[3])
ax.plot(df_index['ret_excess'].unstack()[SPLIT_DATE:]["crsp_vw"], label = "CRSP value-weighted", color=COLORS[1], linestyle="--")
ax.axhline(0, color="k", linestyle=":", linewidth=1)
ax.set_xlim(SPLIT_DATE, df_index['ret_excess'].unstack().index[-1])
add_recession_bars(
    ax, freq="M", startdate=SPLIT_DATE, enddate=df_index['ret_excess'].unstack().index[-1]
)
ax.legend()

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/return.pdf")

# %% [markdown]
# ### Ledoit-Wolf test for diagonal generalized innovations

# %%
# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20, 4))

#
ax1 = ax
l1 = ax1.plot(
    df_stats["innovation_diagonality_test_stat"],
    label="test statistic, mean="
    + str((df_stats["innovation_diagonality_test_stat"]).mean().round(2)),
    c=COLORS[0],
)
ax1.set_ylabel("test statistic", color=COLORS[0])
ax1.tick_params(axis="y", labelcolor=COLORS[0])

#
ax2 = ax1.twinx()
l2 = ax2.plot(
    df_stats["innovation_diagonality_p_value"].astype(float),
    label="p-value, mean="
    + str((df_stats["innovation_diagonality_p_value"]).mean().round(2)),
    c=COLORS[1],
    linestyle="--",
)
ax2.grid(False)
ax2.set_ylim([-0.01, 1.01])
ax2.set_ylabel("p-value", color=COLORS[1])
ax2.tick_params(axis="y", labelcolor=COLORS[1])

# legend
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center left")
add_recession_bars(
    ax, freq="M", startdate=df_stats.index[0], enddate=df_stats.index[-1]
)
ax1.set_xlim([df_stats.index[0], df_stats.index[-1]])

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "rolling/ledoitwolf_test.pdf")

# %% [markdown]
# ### Average factor loading

# %%
df_estimates["var_factor_loadings_crsp"].unstack()[SPLIT_DATE:].mean(axis=1).plot(figsize=(20, 4))
