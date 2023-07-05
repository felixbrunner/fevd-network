# # Crossectional Portfolio Sorts

# ## Imports

# %load_ext autoreload
# %autoreload 2
# # %matplotlib inline

# +
import pandas as pd
import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf
import linearmodels as lm

import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from euraculus.data.map import DataMap
import kungfu as kf

from euraculus.settings import NASDAQ_INCLUSION_DATE, SPLIT_DATE, OUTPUT_DIR

# -

data = DataMap()
df_estimates = data.read("analysis/df_estimates.pkl")


# ## Portfolio sorts


def sort_portfolios(
    df_estimates: pd.DataFrame, variable: "str", num_portfolios: int = 5
):
    """Sort assets into portfolios based on estimates.

    Args:
        df_estimates: Dataframe that contains the ranking variable.
        variable: Name of the ranking variable.
        num_portfolios: The number of portfolios to sort into.

    Returns:
        portfolios: Series with the mapping to the portfolios.
    """
    labels = ["Low"] + list(np.arange(2, num_portfolios).astype(str)) + ["High"]
    portfolios = (
        df_estimates[variable]
        .groupby("sampling_date")
        .apply(lambda x: pd.qcut(x, q=num_portfolios, labels=labels))
        .rename(f"{variable}_portfolio")
    )
    return portfolios


def mean_estimates(df_estimates: pd.DataFrame, portfolios: pd.Series) -> pd.DataFrame:
    """"""
    return (
        df_estimates.reset_index().groupby(["sampling_date", portfolios.values]).mean()
    )


def calculate_portolio_stats(portfolio_estimates, stat: str, transform=None):
    """"""
    # prepare monthly portfolio stats
    df_monthly = portfolio_estimates[stat].unstack()
    df_monthly.columns = df_monthly.columns.values.tolist()
    if transform is not None:
        df_monthly = transform(df_monthly)
    df_monthly["HML"] = df_monthly["High"] - df_monthly["Low"]

    # aggregate
    portfolio_stats = df_monthly.mean()
    # portfolio_stats["t-statistic"] = df_monthly["HML"].mean() / (df_monthly["HML"].std() / df_monthly["HML"].count()**0.5)
    reg = smf.ols("HML ~ 1", data=df_monthly[["HML"]]).fit(
        cov_type="HAC", cov_kwds={"maxlags": round(len(df_monthly) ** 0.25)}
    )
    portfolio_stats["t-statistic"] = reg.tvalues["Intercept"]
    return portfolio_stats


def standard_errors(e: np.ndarray, X: np.ndarray) -> np.ndarray:
    """"""
    N = len(e)
    S0 = 0
    for i in range(N):
        S0 += e[i] ** 2 * (X[i, :].T @ X[i, :])

    return S0


def newey_west_standard_errors(
    e: np.ndarray, X: np.ndarray, lags: int = None
) -> np.ndarray:
    """"""
    T = len(e)
    if lags is None:
        lags = round(T**0.25)

    S0 = standard_errors(e=e, X=X)
    Q = S0

    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        for t in range(l, T):
            ee = e[t] * e[t - l]
            xx = (X[t, :].T @ X[t - 1, :]) + (X[t - l, :].T @ X[t, :])
            # print(1/T * w * ee * xx)
            Q += 1 / T * w * ee * xx  # np.linalg.inv(xx)

    return Q


def make_portfolio_sort_table(df: pd.DataFrame, variable: str, h=12, n_portfolios=5):
    """"""
    portfolios = sort_portfolios(df, variable, n_portfolios)
    portfolio_estimates = mean_estimates(df, portfolios)

    fevd_table = pd.DataFrame(
        columns=["Low", "2", "3", "4", "High", "HML", "t-statistic"]
    )

    rows = {
        "\textit{out}-degree": calculate_portolio_stats(
            portfolio_estimates, "fevd_out_connectedness"
        ),
        "\textit{in}-degree": calculate_portolio_stats(
            portfolio_estimates, "fevd_in_connectedness"
        ),
        "\textit{net}-degree": calculate_portolio_stats(
            portfolio_estimates, "fevd_net_connectedness"
        ),
        "\textit{self}-degree": calculate_portolio_stats(
            portfolio_estimates, "fevd_self_connectedness"
        ),
        "\textit{in}-concentration": calculate_portolio_stats(
            portfolio_estimates, "fevd_in_concentration"
        ),
        "\textit{out}-concentration": calculate_portolio_stats(
            portfolio_estimates, "fevd_out_concentration"
        ),
        # "w \textit{out}-degree": calculate_portolio_stats(portfolio_estimates, "wfevd_out_connectedness"),
        # "w \textit{in}-degree": calculate_portolio_stats(portfolio_estimates, "wfevd_in_connectedness"),
        # "w \textit{net}-degree": calculate_portolio_stats(portfolio_estimates, "wfevd_net_connectedness"),
        # "w \textit{self}-degree": calculate_portolio_stats(portfolio_estimates, "wfevd_self_connectedness"),
        # "w \textit{in}-concentration": calculate_portolio_stats(portfolio_estimates, "wfevd_in_concentration"),
        # "w \textit{out}-concentration": calculate_portolio_stats(portfolio_estimates, "wfevd_out_concentration"),
        "amplification factor": calculate_portolio_stats(
            portfolio_estimates, "fevd_amplification_factor"
        ),
        "absorption rate": calculate_portolio_stats(
            portfolio_estimates, "fevd_absorption_rate"
        ),
        "VARX intercept": calculate_portolio_stats(
            portfolio_estimates, "var_intercept"
        ),
        "VARX factor loading": calculate_portolio_stats(
            portfolio_estimates, "var_factor_loadings_crsp"
        ),
        "VARX average spillover loading": calculate_portolio_stats(
            portfolio_estimates, "var_mean_abs_in"
        ),  # , lambda x: x*99),
        "VARX residual variance": calculate_portolio_stats(
            portfolio_estimates, "cov_variance"
        ),  # , lambda x: np.sqrt(x)),
        "VARX average residual correlation": calculate_portolio_stats(
            portfolio_estimates, "cov_mean_corr"
        ),
        "VARX $R^2$": calculate_portolio_stats(portfolio_estimates, "var_r2"),
        "VARX partial $R^2$: factors": calculate_portolio_stats(
            portfolio_estimates, "var_partial_r2_factors"
        ),
        "VARX partial $R^2$: spillovers": calculate_portolio_stats(
            portfolio_estimates, "var_partial_r2_spillovers"
        ),
        # "VARX systematic variance": calculate_portolio_stats(portfolio_estimates, "var_systematic_variance"),#, lambda x: np.sqrt(x)),
        # "VARX factor residual variance": calculate_portolio_stats(portfolio_estimates, "var_factor_residual_variance"),#, lambda x: np.sqrt(x)),
        # "VARX residual variance": calculate_portolio_stats(portfolio_estimates, "var_residual_variance"),#, lambda x: np.sqrt(x)),
        "$r^e$": calculate_portolio_stats(portfolio_estimates, "ret_excess"),
        "$\sigma(r^e)$": calculate_portolio_stats(
            portfolio_estimates, "var_annual", lambda x: np.sqrt(x)
        ),
        "$s$": calculate_portolio_stats(portfolio_estimates, "xvar_data_mean"),
        "$\sigma(s)$": calculate_portolio_stats(
            portfolio_estimates, "xvar_data_var", lambda x: np.sqrt(x)
        ),
        "$log(\text{Market capitalization})$": calculate_portolio_stats(
            portfolio_estimates, "mean_mcap", lambda x: np.log(x)
        ),
        "CAPM $\beta_M$": calculate_portolio_stats(portfolio_estimates, "capm_mktrf"),
        "CAPM $\alpha_M (annual)$": calculate_portolio_stats(
            portfolio_estimates, "capm_const", lambda x: x * 252
        ),
        "CAPM $\sigma(\varepsilon)$": calculate_portolio_stats(
            portfolio_estimates, "capm_sigma2", lambda x: np.sqrt(x * 252)
        ),
        "CAPM $R^2$": calculate_portolio_stats(portfolio_estimates, "capm_R2"),
        "$r^e$+": calculate_portolio_stats(portfolio_estimates, f"ret_excess_next{h}M"),
        "$\sigma(r^e)$+": calculate_portolio_stats(
            portfolio_estimates, f"var_annual_next{h}M", lambda x: np.sqrt(x)
        ),
        "CAPM $\beta_M$+": calculate_portolio_stats(
            portfolio_estimates, f"capm_mktrf_next{h}M"
        ),
        "CAPM $\alpha$ (annual)+": calculate_portolio_stats(
            portfolio_estimates, f"capm_const_next{h}M", lambda x: x * 252
        ),
        "CAPM $\sigma(\varepsilon)$+": calculate_portolio_stats(
            portfolio_estimates,
            f"capm_sigma2_next{h}M",
            lambda x: np.sqrt(x.replace(np.inf, np.nan) * 252),
        ),
        "CAPM $R^2$+": calculate_portolio_stats(
            portfolio_estimates,
            f"capm_R2_next{h}M",
            lambda x: x.replace(-np.inf, np.nan),
        ),
        #     "3-Factor $\alpha_M (annual)$": calculate_portolio_stats(portfolio_estimates, "ff3_const", lambda x: x*252),
        #     "3-Factor $\sigma^2(\varepsilon)$": calculate_portolio_stats(portfolio_estimates, "ff3_sigma2", lambda x: np.sqrt(x * 252)),
        #     "4-Factor $\alpha_M (annual)$": calculate_portolio_stats(portfolio_estimates, "c4_const", lambda x: x*252),
        #     "4-Factor $\sigma^2(\varepsilon)$": calculate_portolio_stats(portfolio_estimates, "c4_sigma2", lambda x: np.sqrt(x * 252)),
        #     "3-Factor $\alpha_M (annual)$+": calculate_portolio_stats(portfolio_estimates, f"ff3_const_next{h}M", lambda x: x*252),
        #     "3-Factor $\sigma^2(\varepsilon)$+": calculate_portolio_stats(portfolio_estimates, f"ff3_sigma2_next{h}M", lambda x: np.sqrt(x.replace(np.inf, np.nan) * 252)),
        #     "4-Factor $\alpha_M (annual)$+": calculate_portolio_stats(portfolio_estimates, f"c4_const_next{h}M", lambda x: x*252),
        #     "4-Factor $\sigma^2(\varepsilon)$+": calculate_portolio_stats(portfolio_estimates, f"c4_sigma2_next{h}M", lambda x: np.sqrt(x.replace(np.inf, np.nan) * 252)),
    }
    for name, row in rows.items():
        fevd_table = fevd_table.append(row.rename(name))

    fevd_table = fevd_table.rename(
        columns={
            "Low": "Sattelites",
            "High": "Key Assets",
            "HML": "KMS",
        }
    )
    return fevd_table


for variable in (
    "fevd_in_connectedness",
    "fevd_out_connectedness",
    "fevd_net_connectedness",
    "wfevd_in_connectedness",
    "wfevd_out_connectedness",
    "wfevd_net_connectedness",
    "wfevd_full_out_connectedness",
):
    table = make_portfolio_sort_table(
        df=df_estimates[SPLIT_DATE:], h=12, variable=variable, n_portfolios=5
    )
    table.round(4).to_latex(
        buf=OUTPUT_DIR / "temp" / f"{variable}_table.tex",
        multirow=False,
        multicolumn_format="c",
        na_rep="",
        escape=False,
    )


# ### Plots

# +
fig, axes = plt.subplots(9, 1, figsize=(16, 24))

ax = axes[0]
ax.plot(portfolio_estimates["ret_excess"].unstack().mean(), label="ret_excess")
ax.plot(
    portfolio_estimates["ret_excess_next1M"].unstack().mean() * 12,
    label="ret_excess_next1M",
)
ax.plot(
    portfolio_estimates["ret_excess_next3M"].unstack().mean() * 4,
    label="ret_excess_next3M",
)
ax.plot(
    portfolio_estimates["ret_excess_next6M"].unstack().mean() * 2,
    label="ret_excess_next6M",
)
ax.plot(
    portfolio_estimates["ret_excess_next12M"].unstack().mean(),
    label="ret_excess_next12M",
)
ax.plot(
    portfolio_estimates["ret_excess_next24M"].unstack().mean() / 2,
    label="ret_excess_next24M",
)
ax.plot(
    portfolio_estimates["ret_excess_next36M"].unstack().mean() / 3,
    label="ret_excess_next36M",
)
ax.plot(
    portfolio_estimates["ret_excess_next60M"].unstack().mean() / 5,
    label="ret_excess_next60M",
)
ax.set_title("Average annual return")
ax.legend()

ax = axes[1]
ax.plot(portfolio_estimates["var_annual"].unstack().mean(), label="var_annual")
ax.plot(
    portfolio_estimates["var_annual_next1M"].unstack().mean(), label="var_annual_next1M"
)
ax.plot(
    portfolio_estimates["var_annual_next3M"].unstack().mean(), label="var_annual_next3M"
)
ax.plot(
    portfolio_estimates["var_annual_next6M"].unstack().mean(), label="var_annual_next6M"
)
ax.plot(
    portfolio_estimates["var_annual_next12M"].unstack().mean(),
    label="var_annual_next12M",
)
ax.plot(
    portfolio_estimates["var_annual_next24M"].unstack().mean(),
    label="var_annual_next24M",
)
ax.plot(
    portfolio_estimates["var_annual_next36M"].unstack().mean(),
    label="var_annual_next36M",
)
ax.plot(
    portfolio_estimates["var_annual_next60M"].unstack().mean(),
    label="var_annual_next60M",
)
ax.set_title("Average annual variance")
ax.legend()

ax = axes[2]
ax.plot(portfolio_estimates["capm_mktrf"].unstack().mean(), label="capm_mktrf")
ax.plot(
    portfolio_estimates["capm_mktrf_next1M"].unstack().mean(), label="capm_mktrf_next1M"
)
ax.plot(
    portfolio_estimates["capm_mktrf_next3M"].unstack().mean(), label="capm_mktrf_next3M"
)
ax.plot(
    portfolio_estimates["capm_mktrf_next6M"].unstack().mean(), label="capm_mktrf_next6M"
)
ax.plot(
    portfolio_estimates["capm_mktrf_next12M"].unstack().mean(),
    label="capm_mktrf_next12M",
)
ax.plot(
    portfolio_estimates["capm_mktrf_next24M"].unstack().mean(),
    label="capm_mktrf_next24M",
)
ax.plot(
    portfolio_estimates["capm_mktrf_next36M"].unstack().mean(),
    label="capm_mktrf_next36M",
)
ax.plot(
    portfolio_estimates["capm_mktrf_next60M"].unstack().mean(),
    label="capm_mktrf_next60M",
)
ax.set_title("Average CAPM beta")
ax.legend()

ax = axes[3]
ax.plot(portfolio_estimates["capm_const"].unstack().mean() * 252, label="capm_const")
ax.plot(
    portfolio_estimates["capm_const_next1M"].unstack().mean() * 252,
    label="capm_const_next1M",
)
ax.plot(
    portfolio_estimates["capm_const_next3M"].unstack().mean() * 252,
    label="capm_const_next3M",
)
ax.plot(
    portfolio_estimates["capm_const_next6M"].unstack().mean() * 252,
    label="capm_const_next6M",
)
ax.plot(
    portfolio_estimates["capm_const_next12M"].unstack().mean() * 252,
    label="capm_const_next12M",
)
ax.plot(
    portfolio_estimates["capm_const_next24M"].unstack().mean() * 252,
    label="capm_const_next24M",
)
ax.plot(
    portfolio_estimates["capm_const_next36M"].unstack().mean() * 252,
    label="capm_const_next36M",
)
ax.plot(
    portfolio_estimates["capm_const_next60M"].unstack().mean() * 252,
    label="capm_const_next60M",
)
ax.set_title("Average CAPM alpha")
ax.legend()

ax = axes[4]
ax.plot(portfolio_estimates["ff3_const"].unstack().mean() * 252, label="ff3_const")
ax.plot(
    portfolio_estimates["ff3_const_next1M"].unstack().mean() * 252,
    label="ff3_const_next1M",
)
ax.plot(
    portfolio_estimates["ff3_const_next3M"].unstack().mean() * 252,
    label="ff3_const_next3M",
)
ax.plot(
    portfolio_estimates["ff3_const_next6M"].unstack().mean() * 252,
    label="ff3_const_next6M",
)
ax.plot(
    portfolio_estimates["ff3_const_next12M"].unstack().mean() * 252,
    label="ff3_const_next12M",
)
ax.plot(
    portfolio_estimates["ff3_const_next24M"].unstack().mean() * 252,
    label="ff3_const_next24M",
)
ax.plot(
    portfolio_estimates["ff3_const_next36M"].unstack().mean() * 252,
    label="ff3_const_next36M",
)
ax.plot(
    portfolio_estimates["ff3_const_next60M"].unstack().mean() * 252,
    label="ff3_const_next60M",
)
ax.set_title("Average 3-factor alpha")
ax.legend()

ax = axes[5]
ax.plot(portfolio_estimates["c4_const"].unstack().mean() * 252, label="c4_const")
ax.plot(
    portfolio_estimates["c4_const_next1M"].unstack().mean() * 252,
    label="c4_const_next1M",
)
ax.plot(
    portfolio_estimates["c4_const_next3M"].unstack().mean() * 252,
    label="c4_const_next3M",
)
ax.plot(
    portfolio_estimates["c4_const_next6M"].unstack().mean() * 252,
    label="c4_const_next6M",
)
ax.plot(
    portfolio_estimates["c4_const_next12M"].unstack().mean() * 252,
    label="c4_const_next12M",
)
ax.plot(
    portfolio_estimates["c4_const_next24M"].unstack().mean() * 252,
    label="c4_const_next24M",
)
ax.plot(
    portfolio_estimates["c4_const_next36M"].unstack().mean() * 252,
    label="c4_const_next36M",
)
ax.plot(
    portfolio_estimates["c4_const_next60M"].unstack().mean() * 252,
    label="c4_const_next60M",
)
ax.set_title("Average 4-factor alpha")
ax.legend()

ax = axes[6]
ax.plot(
    portfolio_estimates["capm_sigma2"].replace(np.inf, np.nan).unstack().mean() * 252,
    label="capm_sigma2",
)
ax.plot(
    portfolio_estimates["capm_sigma2_next1M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next1M",
)
ax.plot(
    portfolio_estimates["capm_sigma2_next3M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next3M",
)
ax.plot(
    portfolio_estimates["capm_sigma2_next6M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next6M",
)
ax.plot(
    portfolio_estimates["capm_sigma2_next12M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next12M",
)
ax.plot(
    portfolio_estimates["capm_sigma2_next24M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next24M",
)
ax.plot(
    portfolio_estimates["capm_sigma2_next36M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next36M",
)
ax.plot(
    portfolio_estimates["capm_sigma2_next60M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next60M",
)
ax.set_title("Average CAPM idiosyncratic volatility")
ax.legend()

ax = axes[7]
ax.plot(
    portfolio_estimates["ff3_sigma2"].replace(np.inf, np.nan).unstack().mean() * 252,
    label="ff3_sigma2",
)
ax.plot(
    portfolio_estimates["ff3_sigma2_next1M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next1M",
)
ax.plot(
    portfolio_estimates["ff3_sigma2_next3M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next3M",
)
ax.plot(
    portfolio_estimates["ff3_sigma2_next6M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next6M",
)
ax.plot(
    portfolio_estimates["ff3_sigma2_next12M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next12M",
)
ax.plot(
    portfolio_estimates["ff3_sigma2_next24M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next24M",
)
ax.plot(
    portfolio_estimates["ff3_sigma2_next36M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next36M",
)
ax.plot(
    portfolio_estimates["ff3_sigma2_next60M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next60M",
)
ax.set_title("Average 3-factor idiosyncratic volatility")
ax.legend()

ax = axes[8]
ax.plot(
    portfolio_estimates["c4_sigma2"].replace(np.inf, np.nan).unstack().mean() * 252,
    label="c4_sigma2",
)
ax.plot(
    portfolio_estimates["c4_sigma2_next1M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next1M",
)
ax.plot(
    portfolio_estimates["c4_sigma2_next3M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next3M",
)
ax.plot(
    portfolio_estimates["c4_sigma2_next6M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next6M",
)
ax.plot(
    portfolio_estimates["c4_sigma2_next12M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next12M",
)
ax.plot(
    portfolio_estimates["c4_sigma2_next24M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next24M",
)
ax.plot(
    portfolio_estimates["c4_sigma2_next36M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next36M",
)
ax.plot(
    portfolio_estimates["c4_sigma2_next60M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next60M",
)
ax.set_title("Average 4-factor idiosyncratic volatility")
ax.legend()

plt.show()

# +
fig, axes = plt.subplots(9, 1, figsize=(16, 24))

ax = axes[0]
ax.plot(portfolio_estimates["ret_excess"].unstack().mean(), label="ret_excess")
# ax.plot(portfolio_estimates["ret_excess_next1M"].unstack().mean()*12, label="ret_excess_next1M")
ax.plot(
    portfolio_estimates["ret_excess_next3M"].unstack().mean() * 4,
    label="ret_excess_next3M",
)
# ax.plot(portfolio_estimates["ret_excess_next6M"].unstack().mean()*2, label="ret_excess_next6M")
ax.plot(
    portfolio_estimates["ret_excess_next12M"].unstack().mean(),
    label="ret_excess_next12M",
)
# ax.plot(portfolio_estimates["ret_excess_next24M"].unstack().mean()/2, label="ret_excess_next24M")
# ax.plot(portfolio_estimates["ret_excess_next36M"].unstack().mean()/3, label="ret_excess_next36M")
ax.plot(
    portfolio_estimates["ret_excess_next60M"].unstack().mean() / 5,
    label="ret_excess_next60M",
)
ax.set_title("Average annual return")
ax.legend()

ax = axes[1]
ax.plot(portfolio_estimates["var_annual"].unstack().mean(), label="var_annual")
# ax.plot(portfolio_estimates["var_annual_next1M"].unstack().mean(), label="var_annual_next1M")
ax.plot(
    portfolio_estimates["var_annual_next3M"].unstack().mean(), label="var_annual_next3M"
)
# ax.plot(portfolio_estimates["var_annual_next6M"].unstack().mean(), label="var_annual_next6M")
ax.plot(
    portfolio_estimates["var_annual_next12M"].unstack().mean(),
    label="var_annual_next12M",
)
# ax.plot(portfolio_estimates["var_annual_next24M"].unstack().mean(), label="var_annual_next24M")
# ax.plot(portfolio_estimates["var_annual_next36M"].unstack().mean(), label="var_annual_next36M")
ax.plot(
    portfolio_estimates["var_annual_next60M"].unstack().mean(),
    label="var_annual_next60M",
)
ax.set_title("Average annual variance")
ax.legend()

ax = axes[2]
ax.plot(portfolio_estimates["capm_mktrf"].unstack().mean(), label="capm_mktrf")
# ax.plot(portfolio_estimates["capm_mktrf_next1M"].unstack().mean(), label="capm_mktrf_next1M")
ax.plot(
    portfolio_estimates["capm_mktrf_next3M"].unstack().mean(), label="capm_mktrf_next3M"
)
# ax.plot(portfolio_estimates["capm_mktrf_next6M"].unstack().mean(), label="capm_mktrf_next6M")
ax.plot(
    portfolio_estimates["capm_mktrf_next12M"].unstack().mean(),
    label="capm_mktrf_next12M",
)
# ax.plot(portfolio_estimates["capm_mktrf_next24M"].unstack().mean(), label="capm_mktrf_next24M")
# ax.plot(portfolio_estimates["capm_mktrf_next36M"].unstack().mean(), label="capm_mktrf_next36M")
ax.plot(
    portfolio_estimates["capm_mktrf_next60M"].unstack().mean(),
    label="capm_mktrf_next60M",
)
ax.set_title("Average CAPM beta")
ax.legend()

ax = axes[3]
ax.plot(portfolio_estimates["capm_const"].unstack().mean() * 252, label="capm_const")
# ax.plot(portfolio_estimates["capm_const_next1M"].unstack().mean()*252, label="capm_const_next1M")
ax.plot(
    portfolio_estimates["capm_const_next3M"].unstack().mean() * 252,
    label="capm_const_next3M",
)
# ax.plot(portfolio_estimates["capm_const_next6M"].unstack().mean()*252, label="capm_const_next6M")
ax.plot(
    portfolio_estimates["capm_const_next12M"].unstack().mean() * 252,
    label="capm_const_next12M",
)
# ax.plot(portfolio_estimates["capm_const_next24M"].unstack().mean()*252, label="capm_const_next24M")
# ax.plot(portfolio_estimates["capm_const_next36M"].unstack().mean()*252, label="capm_const_next36M")
ax.plot(
    portfolio_estimates["capm_const_next60M"].unstack().mean() * 252,
    label="capm_const_next60M",
)
ax.set_title("Average CAPM alpha")
ax.legend()

ax = axes[4]
ax.plot(portfolio_estimates["ff3_const"].unstack().mean() * 252, label="ff3_const")
# ax.plot(portfolio_estimates["ff3_const_next1M"].unstack().mean()*252, label="ff3_const_next1M")
ax.plot(
    portfolio_estimates["ff3_const_next3M"].unstack().mean() * 252,
    label="ff3_const_next3M",
)
# ax.plot(portfolio_estimates["ff3_const_next6M"].unstack().mean()*252, label="ff3_const_next6M")
ax.plot(
    portfolio_estimates["ff3_const_next12M"].unstack().mean() * 252,
    label="ff3_const_next12M",
)
# ax.plot(portfolio_estimates["ff3_const_next24M"].unstack().mean()*252, label="ff3_const_next24M")
# ax.plot(portfolio_estimates["ff3_const_next36M"].unstack().mean()*252, label="ff3_const_next36M")
ax.plot(
    portfolio_estimates["ff3_const_next60M"].unstack().mean() * 252,
    label="ff3_const_next60M",
)
ax.set_title("Average 3-factor alpha")
ax.legend()

ax = axes[5]
ax.plot(portfolio_estimates["c4_const"].unstack().mean() * 252, label="c4_const")
# ax.plot(portfolio_estimates["c4_const_next1M"].unstack().mean()*252, label="c4_const_next1M")
ax.plot(
    portfolio_estimates["c4_const_next3M"].unstack().mean() * 252,
    label="c4_const_next3M",
)
# ax.plot(portfolio_estimates["c4_const_next6M"].unstack().mean()*252, label="c4_const_next6M")
ax.plot(
    portfolio_estimates["c4_const_next12M"].unstack().mean() * 252,
    label="c4_const_next12M",
)
# ax.plot(portfolio_estimates["c4_const_next24M"].unstack().mean()*252, label="c4_const_next24M")
# ax.plot(portfolio_estimates["c4_const_next36M"].unstack().mean()*252, label="c4_const_next36M")
ax.plot(
    portfolio_estimates["c4_const_next60M"].unstack().mean() * 252,
    label="c4_const_next60M",
)
ax.set_title("Average 3-factor alpha")
ax.legend()

ax = axes[6]
ax.plot(
    portfolio_estimates["capm_sigma2"].replace(np.inf, np.nan).unstack().mean() * 252,
    label="capm_sigma2",
)
# ax.plot(portfolio_estimates["capm_sigma2_next1M"].replace(np.inf, np.nan).unstack().mean()*252, label="capm_sigma2_next1M")
ax.plot(
    portfolio_estimates["capm_sigma2_next3M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next3M",
)
# ax.plot(portfolio_estimates["capm_sigma2_next6M"].replace(np.inf, np.nan).unstack().mean()*252, label="capm_sigma2_next6M")
ax.plot(
    portfolio_estimates["capm_sigma2_next12M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next12M",
)
# ax.plot(portfolio_estimates["capm_sigma2_next24M"].replace(np.inf, np.nan).unstack().mean()*252, label="capm_sigma2_next24M")
# ax.plot(portfolio_estimates["capm_sigma2_next36M"].replace(np.inf, np.nan).unstack().mean()*252, label="capm_sigma2_next36M")
ax.plot(
    portfolio_estimates["capm_sigma2_next60M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="capm_sigma2_next60M",
)
ax.set_title("Average CAPM idiosyncratic volatility")
ax.legend()

ax = axes[7]
ax.plot(
    portfolio_estimates["ff3_sigma2"].replace(np.inf, np.nan).unstack().mean() * 252,
    label="ff3_sigma2",
)
# ax.plot(portfolio_estimates["ff3_sigma2_next1M"].replace(np.inf, np.nan).unstack().mean()*252, label="ff3_sigma2_next1M")
ax.plot(
    portfolio_estimates["ff3_sigma2_next3M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next3M",
)
# ax.plot(portfolio_estimates["ff3_sigma2_next6M"].replace(np.inf, np.nan).unstack().mean()*252, label="ff3_sigma2_next6M")
ax.plot(
    portfolio_estimates["ff3_sigma2_next12M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next12M",
)
# ax.plot(portfolio_estimates["ff3_sigma2_next24M"].replace(np.inf, np.nan).unstack().mean()*252, label="ff3_sigma2_next24M")
# ax.plot(portfolio_estimates["ff3_sigma2_next36M"].replace(np.inf, np.nan).unstack().mean()*252, label="ff3_sigma2_next36M")
ax.plot(
    portfolio_estimates["ff3_sigma2_next60M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="ff3_sigma2_next60M",
)
ax.set_title("Average 3-factor idiosyncratic volatility")
ax.legend()

ax = axes[8]
ax.plot(
    portfolio_estimates["c4_sigma2"].replace(np.inf, np.nan).unstack().mean() * 252,
    label="c4_sigma2",
)
# ax.plot(portfolio_estimates["c4_sigma2_next1M"].replace(np.inf, np.nan).unstack().mean()*252, label="c4_sigma2_next1M")
ax.plot(
    portfolio_estimates["c4_sigma2_next3M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next3M",
)
# ax.plot(portfolio_estimates["c4_sigma2_next6M"].replace(np.inf, np.nan).unstack().mean()*252, label="c4_sigma2_next6M")
ax.plot(
    portfolio_estimates["c4_sigma2_next12M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next12M",
)
# ax.plot(portfolio_estimates["c4_sigma2_next24M"].replace(np.inf, np.nan).unstack().mean()*252, label="c4_sigma2_next24M")
# ax.plot(portfolio_estimates["c4_sigma2_next36M"].replace(np.inf, np.nan).unstack().mean()*252, label="c4_sigma2_next36M")
ax.plot(
    portfolio_estimates["c4_sigma2_next60M"].replace(np.inf, np.nan).unstack().mean()
    * 252,
    label="c4_sigma2_next60M",
)
ax.set_title("Average 4-factor idiosyncratic volatility")
ax.legend()

plt.show()
# -

# ### Autorcorrelation

df_estimates["fevd_out_connectedness"].unstack().asfreq("a").apply(
    lambda x: x.autocorr(1)
).mean()

df_estimates["wfevd_out_connectedness"].unstack().asfreq("a").apply(
    lambda x: x.autocorr(1)
).mean()

df_estimates["wfevd_in_connectedness"].unstack().asfreq("a").apply(
    lambda x: x.autocorr(1)
).mean()

df_estimates["wfevd_net_connectedness"].unstack().asfreq("a").apply(
    lambda x: x.autocorr(1)
).mean()


# +
outcomes = [
    "Total return (t)",
    "Total return (t+1)",
    "beta",
    "beta_t+1",
    "alpha",
    "alpha_t+1",
    "Volatility p.a. (t)",
    "Volatility p.a. (t+1)",
]
rankings = [
    "beta",
    "fev_others",
    "in_connectedness",
    "fev_all",
]  # , 'VAR_intercept', 'in_lvl']

fig, axes = plt.subplots(len(outcomes), len(rankings), figsize=[20, 15])

for i, outcome in enumerate(outcomes):
    for j, ranking in enumerate(rankings):
        axes[i, j].plot(
            mean_portfolio_stat(df, ranking, outcome, 5).mean(),
            marker="o",
            markersize=10,
        )
        axes[i, j].set_title("{} of portfolios sorted on {}".format(outcome, ranking))
