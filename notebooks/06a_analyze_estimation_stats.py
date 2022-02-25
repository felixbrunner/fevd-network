# %% [markdown]
# # Regularized FEVD estimation
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import euraculus
from euraculus.data import DataMap
from euraculus.plot import (
    plot_estimation_summary,
    plot_regularisation_summary,
    plot_network_summary,
)

# %% [markdown]
# ## Load data

# %%
data = DataMap("../data")
df_estimation_stats = data.load_estimation_summary()

# %% [markdown]
# ## Calculations

# %%
df_estimation_stats["mean_shrinkage"] = (
    df_estimation_stats["var_nonzero_shrinkage"]
    + df_estimation_stats["covar_full_shrinkage"]
) / 2
df_estimation_stats["cov_used_df"] = (
    df_estimation_stats["precision_density"] * df_estimation_stats["N"] ** 2
    - df_estimation_stats["N"]
) / 2 + df_estimation_stats["N"]
df_estimation_stats["var_regular_lost_df"] = (
    df_estimation_stats["N"] ** 2 + df_estimation_stats["N"]
)
df_estimation_stats["covar_regular_lost_df"] = (
    df_estimation_stats["N"] * (df_estimation_stats["N"] - 1)
) / 2
df_estimation_stats["var_estimate_share"] = df_estimation_stats[
    "var_regular_lost_df"
] / (
    df_estimation_stats["var_regular_lost_df"]
    + df_estimation_stats["covar_regular_lost_df"]
)
df_estimation_stats["mean_density"] = (
    df_estimation_stats["var_matrix_density"]
    * df_estimation_stats["var_estimate_share"]
    + (1 - df_estimation_stats["var_estimate_share"])
    * df_estimation_stats["precision_density"]
)

# %%
df_estimation_stats.mean().round(2).to_frame()

# %% [markdown]
# ## Plotting

# %%
plot_estimation_summary(
    df_estimation_stats,
    save_path="../reports/figures/estimation_summary.pdf",
)

# %%
plot_regularisation_summary(
    df_estimation_stats,
    save_path="../reports/figures/regularisation_summary.pdf",
)

# %%
plot_network_summary(
    df_estimation_stats,
    save_path="../reports/figures/network_summary.pdf",
)
