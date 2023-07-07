# # Prepare analysis
# ## Imports

# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import datetime as dt
import statsmodels as sm
from euraculus.data.map import DataMap
from euraculus.settings import (
    DATA_DIR,
    FIRST_SAMPLING_DATE,
    LAST_SAMPLING_DATE,
    SPLIT_DATE,
    OUTPUT_DIR,
    COLORS,
    TIME_STEP,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import kungfu as kf
from kungfu.plotting import add_recession_bars
from kungfu.series import FinancialSeries
from kungfu.frame import FinancialDataFrame
from euraculus.utils.plot import save_ax_as_pdf
from euraculus.analysis.index import construct_long_weights, construct_residual_weights, calculate_excess_returns, build_index_portfolio
import pandas_datareader as web

# ## Load & prepare data

save_outputs = False

# %%time
data = DataMap()
df_rf = data.load_rf()
df_estimates = data.read("analysis/df_estimates.pkl")
# df_historic = data.read("analysis/df_historic.pkl")
df_future = data.read("analysis/df_future.pkl")
# df_index = data.read("analysis/df_index.pkl")
df_indices = data.read("analysis/df_daily_indices.pkl")
df_crsp_index = data.read("raw/crsp_index.pkl")
df_crsp_index.index = pd.to_datetime(df_crsp_index.index)
df_factors = data.load_famafrench_factors()

# +
df_fred = pd.DataFrame()

df_fred = df_fred.join(web.fred.FredReader("VIXCLS", start=SPLIT_DATE).read(), how="outer")
df_fred = df_fred.join(web.fred.FredReader("DCOILWTICO", start=SPLIT_DATE).read(), how="outer") # Crude WTI
df_fred = df_fred.join(web.fred.FredReader("TEDRATE", start=SPLIT_DATE).read(), how="outer")

df_fred = df_fred.join(web.fred.FredReader("DFF", start=SPLIT_DATE).read(), how="outer") # fed funds effective rate
df_fred = df_fred.join(web.fred.FredReader("DGS3MO", start=SPLIT_DATE).read(), how="outer")
df_fred = df_fred.join(web.fred.FredReader("DGS2", start=SPLIT_DATE).read(), how="outer")
df_fred = df_fred.join(web.fred.FredReader("DGS10", start=SPLIT_DATE).read(), how="outer")
df_fred = df_fred.join(web.fred.FredReader("DGS30", start=SPLIT_DATE).read(), how="outer")

# +
df_fred["vix_ret"] = df_fred["VIXCLS"].pct_change()
df_fred["oil_ret"] = df_fred["DCOILWTICO"].pct_change()
df_fred["ted_ret"] = df_fred["TEDRATE"].pct_change()

df_fred["dff_diff"] = df_fred["DFF"].diff()
df_fred["3m_diff"] = df_fred["DGS3MO"].diff()
df_fred["2y_diff"] = df_fred["DGS2"].diff()
df_fred["10y_diff"] = df_fred["DGS10"].diff()
df_fred["30y_diff"] = df_fred["DGS30"].diff()

df_fred["2y3m_diff"] = (df_fred["DGS2"]-df_fred["DGS3MO"]).diff()
df_fred["10y2y_diff"] = (df_fred["DGS10"]-df_fred["DGS2"]).diff()
df_fred["30y10y_diff"] = (df_fred["DGS30"]-df_fred["DGS10"]).diff()


# -

# ## Granular Indices

def construct_granular_index(
    df_observations: pd.DataFrame,
    df_weighting: pd.DataFrame,
    observation_column: str = "retadj",
    weighting_column: str = "mean_mcap",
    benchmark: str = "equal",
    constant_leverage: bool = False,
    leg: str = "both"
):
    """"""
    weights = construct_residual_weights(df_weighting, variable=weighting_column, benchmark_variable=benchmark, constant_leverage=constant_leverage)
    granular_index = build_index_portfolio(weights, df_observations[[observation_column, "sampling_date"]], leg=leg).rename(f"gi_{weighting_column}_{benchmark}")
    return granular_index


def construct_granular_index_hedged(
    df_observations: pd.DataFrame,
    df_weighting: pd.DataFrame,
    observation_column: str = "retadj",
    hedge_column: str = "capm_mktrf",
    weighting_column: str = "mean_mcap",
    benchmark: str = "equal",
    constant_leverage: bool = False,
    leg: str = "both"
):
    """"""
    weights = construct_residual_weights(df_weighting, variable=weighting_column, benchmark_variable=benchmark, constant_leverage=constant_leverage)
    beta_contributions = weights * df_weighting[hedge_column]
    print(beta_contributions[SPLIT_DATE:].sum(), beta_contributions[SPLIT_DATE:].sum()/df_estimates[SPLIT_DATE:].index.get_level_values("sampling_date").nunique())
    
    # beta_factor = beta_contributions[weights>0].groupby("sampling_date").sum() / -beta_contributions[weights<0].groupby("sampling_date").sum()
    beta_factor = beta_contributions[SPLIT_DATE:][weights>0].sum() / -beta_contributions[SPLIT_DATE:][weights<0].sum()
    print(beta_factor)
    
    weights[weights>0] = weights[weights>0] / beta_factor
    granular_index = build_index_portfolio(weights, df_observations[[observation_column, "sampling_date"]], leg=leg).rename(f"gi_{weighting_column}_{benchmark}")
    return granular_index


def construct_shock_proxies(
    df_observations: pd.DataFrame,
    df_weighting: pd.DataFrame,
    observation_column: str = "retadj",
    weighting_column: str = "mean_mcap",
    benchmark: str = "equal",
    constant_leverage: bool = False,
    num_proxies: int = None,
):
    """"""
    weights = construct_residual_weights(df_weighting, variable=weighting_column, benchmark_variable=benchmark, constant_leverage=constant_leverage)
    weight_ranks = weights.groupby("sampling_date").rank(ascending=False).rename("weight_ranks")
    
    df_combined = (
        df_future[[observation_column, "sampling_date"]].fillna(0).reset_index()
        .merge(weight_ranks.reset_index(), on=["sampling_date", "permno"], how="inner")
        .set_index(["date", "permno"])
        .drop(columns=["sampling_date"])
    )

    df_proxies = pd.DataFrame()
    for i in range(num_proxies if num_proxies != None else int(weight_ranks.max())):
        df_proxies[i+1] = df_combined[df_combined["weight_ranks"] == i+1].reset_index().set_index("date")[observation_column]
    
    return df_proxies



exposure = beta_contributions.groupby("sampling_date").sum()[SPLIT_DATE:]

exposure.mean()

exposure.std()

exposure.mean()/(exposure.std()/len(exposure)**0.5)



def construct_granular_residual_index(
    df_observations: pd.DataFrame,
    df_weighting: pd.DataFrame,
    observation_column: str = "retadj",
    weighting_column: str = "mean_mcap",
    benchmark: str = "equal",
    constant_leverage: bool = False,
    leg: str = "both"
):
    """"""
    weights = construct_residual_weights(df_weighting, variable=weighting_column, benchmark_variable=benchmark, constant_leverage=constant_leverage)
    weights[weights<0] = 0
    granular_index = build_index_portfolio(weights, df_observations[[observation_column, "sampling_date"]], leg=leg).rename(f"gi_{weighting_column}_{benchmark}")
    return granular_index


def construct_residual_index(
    df_observations: pd.DataFrame,
    df_weighting: pd.DataFrame,
    observation_column: str = "capm_resid",
    weighting_column: str = "mean_mcap",
):
    """"""
    weights = construct_long_weights(df_weighting, variable=weighting_column)
    # weights = df_weighting[weighting_column]/100
    residual_index = build_index_portfolio(weights, df_observations[[observation_column, "sampling_date"]]).rename(f"ri_{weighting_column}")
    return residual_index


df_analysis = df_indices.copy()
df_analysis["ret_vw"] = df_analysis["ret_vw"] -  df_analysis["rf"]
df_analysis = df_analysis.join(construct_granular_index(df_observations=df_future,
                                                        df_weighting=df_estimates,
                                                        observation_column="retadj",
                                                        weighting_column="mean_mcap",
                                                        benchmark="equal",
                                                        constant_leverage=False,
                                                        leg="both",).rename("granular_residual"))
df_analysis = df_analysis.join(construct_granular_index(df_observations=df_future,
                                                        df_weighting=df_estimates,
                                                        observation_column="retadj",
                                                        weighting_column="wfevd_out_connectedness",
                                                        benchmark="equal",
                                                        constant_leverage=False,
                                                        leg="both",).rename("network_residual"))
df_analysis = df_analysis.join(construct_granular_index(df_observations=df_future,
                                                        df_weighting=df_estimates,
                                                        observation_column="capm_future_alphaerror",
                                                        weighting_column="mean_mcap",
                                                        benchmark="equal",
                                                        constant_leverage=False,
                                                        leg="both",).rename("granular_residual_a"))
df_analysis = df_analysis.join(construct_granular_index(df_observations=df_future,
                                                        df_weighting=df_estimates,
                                                        observation_column="capm_future_alphaerror",
                                                        weighting_column="wfevd_out_connectedness",
                                                        benchmark="equal",
                                                        constant_leverage=False,
                                                        leg="both",).rename("network_residual_a"))
df_analysis = df_analysis.join(construct_residual_index(df_observations=df_future,
                                                        df_weighting=df_estimates,
                                                        observation_column="capm_future_alphaerror",
                                                        weighting_column="mean_mcap").rename("granular_residual_index"))
df_analysis = df_analysis.join(construct_residual_index(df_observations=df_future,
                                                        df_weighting=df_estimates,
                                                        observation_column="capm_future_alphaerror",
                                                        weighting_column="wfevd_out_connectedness").rename("network_residual_index"))
df_analysis = df_analysis.join(df_factors, rsuffix="_")

df_residuals = df_analysis[[
    "granular_residual", "network_residual",
    # "granular_residual_a", "network_residual_a",
    "granular_residual_index", "network_residual_index",
    # "granular_residual_beta", "network_residual_beta",
]][SPLIT_DATE:].dropna()

# +
fig, ax = plt.subplots(1, 1, figsize=(16, 4))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylim([-0.025, 0.025])
ax.plot(df_residuals["granular_residual"], label=f"Granular residual index")
add_recession_bars(
    ax, startdate=SPLIT_DATE, enddate=df_residuals.index[-1]
)
ax.set_xlim([df_residuals.index[0], df_residuals.index[-1]])
ax.legend()

# ax2 = ax.twinx()
# ax2.plot(df_residuals["granular_residual"].rolling(21).std()*np.sqrt(252), color=COLORS[1], linestyle=":")
# ax2.grid(False)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "analysis" / "granular_residual_return.pdf")

# +
fig, ax = plt.subplots(1, 1, figsize=(16, 4))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.set_ylim([-0.025, 0.025])
ax.plot(df_residuals["network_residual"], label=f"Network residual index")
add_recession_bars(
    ax, startdate=SPLIT_DATE, enddate=df_residuals.index[-1]
)
ax.set_xlim([df_residuals.index[0], df_residuals.index[-1]])
ax.legend()

# ax2 = ax.twinx()
# ax2.plot(df_residuals["granular_residual"].rolling(21).std()*np.sqrt(252), color=COLORS[1], linestyle=":")
# ax2.grid(False)

if save_outputs:
    save_ax_as_pdf(ax, save_path=OUTPUT_DIR / "analysis" / "network_residual_return.pdf")
# -

df_fin = kf.frame.FinancialDataFrame(df_residuals)
df_fin.obstypes= ["return", "return"]
df_fin.summarise_performance(annual_obs=252)

df_residuals.corr()


# ## Granular OLS regressions

def run_granular_ols(endog: str, exog: list, controls: list = [], frequency: str = "d"):
    """"""
    df_reg=df_analysis[SPLIT_DATE:][[endog]+exog+controls].dropna()
    if frequency == "m":
        df_reg = df_reg.groupby(pd.Grouper(freq="M")).apply(lambda x: (1+x).prod()-1)
        
    reg = sm.regression.linear_model.OLS(
        endog=df_reg[endog],
        exog=sm.tools.tools.add_constant(df_reg[exog+controls]).shift(0).fillna(0),
    ).fit(
        cov_type="HAC",
        cov_kwds={'maxlags': round(len(df_reg)**(1/4))},
    )
    return reg


def make_granular_ols_table(endog: str, exogs: list, controls: list = []):

    reg_table = kf.RegressionTable()
    
    reg = run_granular_ols(endog=endog, exog=exogs, controls=controls, frequency="d")
    reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"])
    
    for exog in exogs:
        reg = run_granular_ols(endog=endog, exog=[exog], controls=controls, frequency="d")
        reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"])
        
    reg = run_granular_ols(endog=endog, exog=exogs, controls=controls, frequency="m")
    reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"])
    
    for exog in exogs:
        reg = run_granular_ols(endog=endog, exog=[exog], controls=controls, frequency="m")
        reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"])

    return reg_table


reg_table = make_granular_ols_table(endog="ret_vw", exogs=["granular_residual", "network_residual"])
reg_table

reg_table = reg_table.change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
    }
).drop_second_index()
if save_outputs:
    reg_table.export_to_latex(str(OUTPUT_DIR / "analysis" / "granular_regressions_ols.tex"))
reg_table

reg_table = make_granular_ols_table(endog="ret_vw", exogs=["granular_residual_index", "network_residual_index"])

reg_table = reg_table.change_row_labels(
    {
        "const": "intercept",
        "granular_residual_index": "granular residual index $\eta^{g}$",
        "network_residual_index": "network residual index $\eta^{nw}$",
    }
).drop_second_index()
if save_outputs:
    reg_table.export_to_latex(str(OUTPUT_DIR / "analysis" / "granular_index_regressions_ols.tex"))
reg_table

df_analysis[[
    "granular_residual",
    "network_residual",
    "granular_residual_index",
    "network_residual_index",
]].corr()

# ## Granular regressions with controls

df_analysis = df_analysis.join(df_fred.iloc[:, 9:])
control_table = kf.RegressionTable()

reg_table = make_granular_ols_table(
    endog="ret_vw",
    exogs=["granular_residual", "network_residual"],
    controls=["dxy_ret", "tnx_ret", "oil_ret"],
).change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
        "dxy_ret": "DXY U.S. Dollar index",
        "tnx_ret": "TNX 10 year treasury yield index",
        "oil_ret": "WTI Crude oil prices",
    }
).drop_second_index()
control_table = control_table.append(reg_table)

reg_table = make_granular_ols_table(
    endog="ret_vw",
    exogs=["granular_residual", "network_residual"],
    controls=["vix_ret", "ted_ret"],
).change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
        "ted_ret": "TED spread",
        "vix_ret": "VIX uncertainty index",
    }
).drop_second_index()
control_table = control_table.append(reg_table)

reg_table = make_granular_ols_table(
    endog="ret_vw",
    exogs=["granular_residual", "network_residual"],
    controls=['dff_diff', '3m_diff', '2y_diff', '10y_diff', '30y_diff'],
).change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
        "dff_diff": "Fed funds effective rate",
        "3m_diff": "3-month treasury yield",
        "2y_diff": "2-year treasury yield",
        "10y_diff": "10-year treasury yield",
        "30y_diff": "30-year treasury yield",
    }
).drop_second_index()
control_table = control_table.append(reg_table)

reg_table = make_granular_ols_table(
    endog="ret_vw",
    exogs=["granular_residual", "network_residual"],
    controls=['2y3m_diff', '10y2y_diff', '30y10y_diff'],
).change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
        "2y3m_diff": "2y-3m yield slope",
        "10y2y_diff": "10y-2y yield slope",
        "30y10y_diff": "30y-10y yield slope",
    }
).drop_second_index()
control_table = control_table.append(reg_table)

reg_table = make_granular_ols_table(
    endog="ret_vw",
    exogs=["granular_residual", "network_residual"],
    controls=["smb", "hml", "umd"],
).change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
        "smb": "SMB",
        "hml": "HML",
        "umd": "UMD",
    }
).drop_second_index()
control_table = control_table.append(reg_table)

reg_table = make_granular_ols_table(
    endog="ret_vw",
    exogs=["granular_residual", "network_residual"],
    controls=[
        "dxy_ret", "tnx_ret", "oil_ret",
        "vix_ret", "ted_ret",
        'dff_diff', '3m_diff', '2y_diff', '10y_diff', '30y_diff',
        '2y3m_diff', '10y2y_diff', '30y10y_diff',
        "smb", "hml", "umd",
    ],
).change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
        "dxy_ret": "DXY U.S. Dollar index",
        "tnx_ret": "TNX 10 year treasury yield index",
        "oil_ret": "WTI Crude oil prices",
        "ted_ret": "TED spread",
        "vix_ret": "VIX uncertainty index",
        "dff_diff": "Fed funds effective rate",
        "3m_diff": "3-month treasury yield",
        "2y_diff": "2-year treasury yield",
        "10y_diff": "10-year treasury yield",
        "30y_diff": "30-year treasury yield",
        "2y3m_diff": "2y-3m yield slope",
        "10y2y_diff": "10y-2y yield slope",
        "30y10y_diff": "30y-10y yield slope",
        "smb": "SMB",
        "hml": "HML",
        "umd": "UMD",
    }
).drop_second_index()
control_table = control_table.append(reg_table)

if save_outputs:
    control_table.export_to_latex(str(OUTPUT_DIR / "analysis" / "granular_regressions_controls.tex"))
control_table

# ## Granular IV regressions

from euraculus.models.factor import CAPM
from statsmodels.sandbox.regression.gmm import IV2SLS
from tqdm import tqdm


class FullIV:
    
    def __init__(
        self,
        endog: pd.DataFrame,
        exog: pd.DataFrame,
        instruments=None,
    ):
        """"""
        self.endog = endog
        self.exog = exog
        self.instruments = instruments
        
    @property
    def exog_names(self):
        return self.exog.columns
    
    @property
    def instrument_names(self):
        return self.instruments.columns
    
    @property
    def y(self):
        return self.endog.values
    
    @property
    def X(self):
        return self.exog.values
    
    @property
    def Z(self):
        return self.instruments.values
    
    @property
    def nobs(self):
        return len(self.endog)
        
    def fit(self, C: float = 1):
        """"""
        # set up
        y = self.y
        X = self.X
        Z = self.Z
        T = self.nobs
        K = self.Z.shape[1]
        G = self.X.shape[1]
    
        P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T

        X_bar = np.concatenate([y, X], axis=1)
        M = np.linalg.inv(X_bar.T @ X_bar) @ (X_bar.T @ P @ X_bar)
        alpha_tilde = min(np.linalg.eigvals(M))
        alpha_hat = (alpha_tilde - (1-alpha_tilde) * C / T) / (1 - (1-alpha_tilde) * C / T)
        # alpha_hat = alpha_tilde #LIML estimator

        coef_ = np.linalg.inv(X.T @ P @ X - alpha_hat * X.T @ X) @ (X.T @ P @ y - alpha_hat * X.T @ y)

        u_hat = y - X @ coef_
        sigma2_hat = u_hat.T @ u_hat / (T-G)
        alpha_tilde = u_hat.T @ P @ u_hat / (u_hat.T @ u_hat)
        Gamma = P @ X
        X_tilde = X - u_hat @ (u_hat.T @ X) / (u_hat.T @ u_hat)
        V_hat = (np.eye(P.shape[0]) - P) @ X_tilde
        kappa = (np.diag(P)**2 / K).sum()
        tau = K / T
        H_hat = X.T @ P @ X - alpha_tilde * X.T @ X

        SigmaB_hat = sigma2_hat * ((1-alpha_tilde)**2 * X_tilde.T @ P @ X_tilde + alpha_tilde**2 * X_tilde.T @ (np.eye(P.shape[0]) - P) @ X_tilde)

        A_hat = 0
        B_hat = 0
        A1 = np.diag(np.diag(P) - tau) @ Gamma
        A2 = (u_hat.T**2 @ V_hat / T)
        B1 = K * (kappa-tau) / (T - (1-2*tau + kappa * tau))
        for t in tqdm(range(T)):
            A_hat += A1[[t]].T @ A2
            B_hat = B1 * (u_hat[t] - sigma2_hat) * V_hat[[t]].T @ V_hat[[t]]
        Sigma_hat = SigmaB_hat + A_hat + A_hat.T + B_hat

        Lambda_hat = np.linalg.inv(H_hat) @ Sigma_hat @ np.linalg.inv(H_hat)
        
        self.params = pd.Series(index=self.exog_names, data=coef_.squeeze())
        self.bse = pd.Series(index=self.exog_names, data=np.diag(Lambda_hat)**0.5)
        self.tvalues = self.params / self.bse
        
        return self
    
    @property
    def tss(self):
        return ((self.y - self.y.mean())**2).sum()
    
    @property
    def fitted_values(self):
        return self.X @ self.params.values
    
    @property
    def residuals(self):
        return self.y.squeeze() - self.fitted_values
    
    @property
    def rss(self):
        return (self.residuals**2).sum()
    
    @property
    def rsquared(self):
        return 1 - self.rss/self.tss
    
    def _create_summary_column(
        self,
        t_stats: bool = True,
        add_outputs: list = [],
    ):
        """"""
        summary = pd.Series(
            index=pd.MultiIndex.from_product(
                [self.params.index, ["coeff", "t-stat"]]
            )
        )
        stars = (
            (abs(self.tvalues) > 1.645).astype(int)
            + (abs(self.tvalues) > 1.96).astype(int)
            + (abs(self.tvalues) > 2.58).astype(int)
        )
        summary.loc[(self.params.index, "coeff")] = (
            self.params.map(lambda x: "%.4f" % x).values
            + stars.map(lambda x: x * "*").values
        )

        if t_stats:
            summary.loc[(self.params.index, "t-stat")] = self.tvalues.map(
                lambda x: "(%.4f)" % x
            ).values
        else:
            summary.index = pd.MultiIndex.from_product(
                [self.params.index, ["coeff", "s.e."]]
            )
            summary.loc[(self.params.index, "s.e.")] = self.bse.map(
                lambda x: "(%.4f)" % x
            ).values

        output_dict = {
            "R-squared": "self.rsquared",
            "N": "self.nobs",
            # "Adj R-squared": "regression.rsquared_adj",
            # "AIC": "regression.aic",
            # "BIC": "regression.bic",
            # "LL": "regression.llf",
            # "F-stat": "regression.fvalue",
            # "P(F-stat)": "regression.f_pvalue",
            # "DF (model)": "regression.df_model",
            # "DF (residuals)": "regression.df_resid",
            # "MSE (model)": "regression.mse_model",
            # "MSE (residuals)": "regression.mse_resid",
            # "MSE (total)": "regression.mse_total",
        }

        for out in add_outputs:
            if out in ["N", "DF (model)", "DF (residuals)"]:
                summary[(out, "")] = "{:.0f}".format(eval(output_dict[out]))
            else:
                try:
                    summary[(out, "")] = "{:.4f}".format(eval(output_dict[out]))
                except:
                    pass

        return summary


def run_granular_iv(endog: str, exog: list, instrument: str = "capm_future_error", frequency: str = "d", estimator: str="2sls"):
    """"""
    df_reg=df_analysis[SPLIT_DATE:][[endog]+exog].dropna()
    df_iv = df_future[instrument][df_reg.index].unstack()
    df_iv = sm.tools.add_constant(df_iv.fillna(df_iv.mean()))
    df_iv = df_iv.loc[:, df_iv.isna().sum()==0]
    if frequency == "m":
        df_reg = df_reg.groupby(pd.Grouper(freq="M")).apply(lambda x: (1+x).prod()-1)
        df_iv = df_iv.groupby(pd.Grouper(freq="M")).apply(lambda x: (1+x).prod()-1)
        # df_reg = df_reg.groupby(pd.Grouper(freq="M")).apply(lambda x: np.log(1+x).sum())
        # df_iv = df_iv.groupby(pd.Grouper(freq="M")).apply(lambda x: x.sum())
    if estimator == "2sls":
        reg = sm.sandbox.regression.gmm.IV2SLS(
            endog=df_reg[endog],
            exog=sm.tools.tools.add_constant(df_reg[exog]).shift(0).fillna(0),
            instrument=df_iv,
        ).fit()
    elif estimator == "full":
        reg = FullIV(
            endog=df_reg[[endog]],
            exog=sm.tools.tools.add_constant(df_reg[exog]).shift(0).fillna(0),
            instruments=df_iv,
        ).fit()
    return reg


def make_granular_iv_table(endog: str, exogs: list, instrument: str = "capm_future_error", estimator: str="2sls", t_stats=True):

    if estimator=="2sls":
        reg_table = kf.RegressionTable()
    
        reg = run_granular_iv(endog=endog, exog=exogs, instrument=instrument, frequency="d", estimator=estimator)
        reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"], t_stats=t_stats)
    
        for exog in exogs:
            reg = run_granular_iv(endog=endog, exog=[exog], instrument=instrument, frequency="d", estimator=estimator)
            reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"], t_stats=t_stats)

        reg = run_granular_iv(endog=endog, exog=exogs, instrument=instrument, frequency="m", estimator=estimator)
        reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"], t_stats=t_stats)

        for exog in exogs:
            reg = run_granular_iv(endog=endog, exog=[exog], instrument=instrument, frequency="m", estimator=estimator)
            reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"], t_stats=t_stats)
    
    if estimator=="full":
        count = 1
        reg = run_granular_iv(endog=endog, exog=exogs, instrument=instrument, frequency="d", estimator=estimator)
        reg_table = kf.RegressionTable(reg._create_summary_column(add_outputs=["R-squared", "N"], t_stats=t_stats).rename(f"({count})"))
        count += 1
    
        for exog in exogs:
            reg = run_granular_iv(endog=endog, exog=[exog], instrument=instrument, frequency="d", estimator=estimator)
            reg_table = reg_table.join(reg._create_summary_column(add_outputs=["R-squared", "N"], t_stats=t_stats).rename(f"({count})"))
            count += 1

        reg = run_granular_iv(endog=endog, exog=exogs, instrument=instrument, frequency="m", estimator=estimator)
        reg_table = reg_table.join(reg._create_summary_column(add_outputs=["R-squared", "N"], t_stats=t_stats).rename(f"({count})"))
        count += 1
                    
        for exog in exogs:
            reg = run_granular_iv(endog=endog, exog=[exog], instrument=instrument, frequency="m", estimator=estimator)
            reg_table = reg_table.join(reg._create_summary_column(add_outputs=["R-squared", "N"], t_stats=t_stats).rename(f"({count})"))
            count += 1

    return reg_table.replace(np.nan, "")


estimator_table = kf.RegressionTable()

iv_table = make_granular_iv_table(endog="ret_vw", exogs=["granular_residual", "network_residual"], instrument="capm_future_error", estimator="2sls", t_stats=True)
iv_table.change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
    }
).drop_second_index()
estimator_table = estimator_table.append(iv_table)
iv_table

iv_table = make_granular_iv_table(endog="ret_vw", exogs=["granular_residual", "network_residual"], instrument="capm_future_error", estimator="full", t_stats=True)
iv_table.change_row_labels(
    {
        "const": "intercept",
        "granular_residual": "granular residual $\eta^{g}$",
        "network_residual": "network residual $\eta^{nw}$",
    }
).drop_second_index()
estimator_table = estimator_table.append(iv_table)
iv_table

if save_outputs:
    estimator_table.export_to_latex(str(OUTPUT_DIR / "analysis" / "granular_regressions_iv.tex"))



# +
# first_stage = sm.regression.linear_model.OLS(endog=df_reg["network_residual"], exog=df_iv).fit()
# first_stage.summary()

# +
# zero_stage = sm.regression.linear_model.OLS(endog=df_reg["ret_vw"], exog=df_iv).fit()
# zero_stage.summary()
# -





def run_granular_iv(endog: str, exog: list, instrument: str = "capm_future_error", frequency: str = "d", estimator: str="2sls", num_proxies=100):
    """"""
    df_reg=df_analysis[SPLIT_DATE:][[endog]+exog].dropna()
    df_proxies = construct_shock_proxies(
        df_observations=df_future,
        df_weighting=df_estimates,
        observation_column="capm_future_error",
        weighting_column="wfevd_out_connectedness",
        benchmark="equal",
        num_proxies=num_proxies,
    )
    df_reg = df_reg.merge(df_proxies, how="left", left_index=True, right_index=True)
    df_iv = df_reg.loc[:,[type(c)==int for c in df_reg.columns]]
    df_iv = sm.tools.add_constant(df_iv.fillna(0))
    # df_iv = df_iv.loc[:, df_iv.isna().sum()==0]
    if frequency == "m":
        df_reg = df_reg.groupby(pd.Grouper(freq="M")).apply(lambda x: (1+x).prod()-1)
        df_iv = df_iv.groupby(pd.Grouper(freq="M")).apply(lambda x: (1+x).prod()-1)
        # df_reg = df_reg.groupby(pd.Grouper(freq="M")).apply(lambda x: np.log(1+x).sum())
        # df_iv = df_iv.groupby(pd.Grouper(freq="M")).apply(lambda x: x.sum())
    if estimator == "2sls":
        reg = sm.sandbox.regression.gmm.IV2SLS(
            endog=df_reg[endog],
            exog=sm.tools.tools.add_constant(df_reg[exog]).shift(0).fillna(0),
            instrument=df_iv,
        ).fit()
    elif estimator == "full":
        reg = FullIV(
            endog=df_reg[[endog]],
            exog=sm.tools.tools.add_constant(df_reg[exog]).shift(0).fillna(0),
            instruments=df_iv,
        ).fit()
    return reg


make_granular_iv_table(endog="ret_vw", exogs=["granular_residual", "network_residual"], instrument="capm_future_error", estimator="2sls", t_stats=True)





def hful_iv(y, X, Z, C=1):
    
    N = len(y)
    K = Z.shape[1]
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    
    X_bar = np.concatenate([y, X], axis=1)
    M = np.linalg.inv(X_bar.T @ X_bar) @ (X_bar.T @ (P - np.diag(np.diag(P))) @ X_bar)
    alpha_tilde = min(np.linalg.eigvals(M))
    alpha_hat = (alpha_tilde - (1-alpha_tilde) * C / N) / (1 - (1-alpha_tilde) * C / N)
    
    coef_ = np.linalg.inv(X.T @ (P - np.diag(np.diag(P))) @ X - alpha_hat * X.T @ X) @ (X.T @ (P - np.diag(np.diag(P))) @ y - alpha_hat * X.T @ y)
    
    epsilon_hat = y - X @ coef_
    gamma_hat = X.T @ epsilon_hat / (epsilon_hat.T @ epsilon_hat)
    X_hat = X - epsilon_hat @ gamma_hat.T
    X_dot = P @ X_hat
    Z_tilde = Z @ np.linalg.inv(Z.T @ Z)
    H_hat = X.T @ (P - np.diag(np.diag(P))) @ X - alpha_hat * X.T @ X
    
    sum1 = 0
    for i in range(N):
        sum1 += (X_dot[[i]].T @ X_dot[[i]] - X_hat[[i]].T @ np.array([[P[i,i]]]) @ X_dot[[i]] - X_dot[[i]].T @ np.array([[P[i,i]]]) @ X_hat[[i]]) * epsilon_hat[[i]]**2
    sum2 = 0
    for k in tqdm(range(K), "calculating covariance"):
        for l in range(K):
            isum1 = (Z_tilde[:,[k]] * Z_tilde[:,[l]] * epsilon_hat).T @ X_hat
            isum2 = (Z[:,[k]] * Z[:,[l]] * epsilon_hat).T @ X_hat
            sum2 += isum1 @ isum2.T    
    Sigma_hat = sum1 + sum2
    
    V_hat = np.linalg.inv(H_hat) @ Sigma_hat @ np.linalg.inv(H_hat)
    
    return coef_, V_hat


y=df_reg[["ret_vw"]].values
X=sm.tools.tools.add_constant(df_reg[[
    "granular_residual", 
    "network_residual",
]]).values
Z=df_iv.values
C=1

coef_, coef_var_ = hful_iv(
    y=df_reg[["ret_vw"]].values, 
    X=sm.tools.tools.add_constant(df_reg[[
        "granular_residual", 
        "network_residual",
    ]]).values, 
    Z=df_iv.values)

t_ratios_ = coef_.squeeze() / np.diag(coef_var_)**0.5
coef_.squeeze(), t_ratios_



def full_iv(y, X, Z, C=1):
    
    T = len(y)
    K = Z.shape[1]
    G = X.shape[1]
    
    P = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    
    X_bar = np.concatenate([y, X], axis=1)
    M = np.linalg.inv(X_bar.T @ X_bar) @ (X_bar.T @ P @ X_bar)
    alpha_tilde = min(np.linalg.eigvals(M))
    alpha_hat = (alpha_tilde - (1-alpha_tilde) * C / T) / (1 - (1-alpha_tilde) * C / T)
    
    coef_ = np.linalg.inv(X.T @ P @ X - alpha_hat * X.T @ X) @ (X.T @ P @ y - alpha_hat * X.T @ y)
    
    u_hat = y - X @ coef_
    sigma2_hat = u_hat.T @ u_hat / (T-G)
    alpha_tilde = u_hat.T @ P @ u_hat / (u_hat.T @ u_hat)
    Gamma = P @ X
    X_tilde = X - u_hat @ (u_hat.T @ X) / (u_hat.T @ u_hat)
    V_hat = (np.eye(P.shape[0]) - P) @ X_tilde
    kappa = (np.diag(P)**2 / K).sum()
    tau = K / T
    H_hat = X.T @ P @ X - alpha_tilde * X.T @ X
    
    SigmaB_hat = sigma2_hat * ((1-alpha_tilde)**2 * X_tilde.T @ P @ X_tilde + alpha_tilde**2 * X_tilde.T @ (np.eye(P.shape[0]) - P) @ X_tilde)
    
    A_hat = 0
    B_hat = 0
    A1 = np.diag(np.diag(P) - tau) @ Gamma
    A2 = (u_hat.T**2 @ V_hat / T)
    B1 = K * (kappa-tau) / (T - (1-2*tau + kappa * tau))
    for t in tqdm(range(T)):
        A_hat += A1[[t]].T @ A2
        B_hat = B1 * (u_hat[t] - sigma2_hat) * V_hat[[t]].T @ V_hat[[t]]
    Sigma_hat = SigmaB_hat + A_hat + A_hat.T + B_hat
    
    Lambda_hat = np.linalg.inv(H_hat) @ Sigma_hat @ np.linalg.inv(H_hat)
    
    return coef_, Lambda_hat


y=df_reg[["ret_vw"]].values
X=sm.tools.tools.add_constant(df_reg[[
    "granular_residual", 
    "network_residual",
]]).values
Z=df_iv.values
C=1

coef_, coef_var_ = full_iv(
    y=df_reg[["ret_vw"]].values, 
    X=sm.tools.tools.add_constant(df_reg[[
        "granular_residual", 
        "network_residual",
    ]]).values, 
    Z=df_iv.values)

se_ = np.diag(coef_var_)**0.5
t_ratios_ = coef_.squeeze() / se_
coef_.squeeze(), t_ratios_





from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

components = PCA(n_components=20).fit_transform(df_iv)

factors = FactorAnalysis(n_components=20).fit_transform(df_iv)

_ = PCA(n_components=50).fit(df_iv)

pd.Series(_.explained_variance_ratio_.cumsum()).plot()

_ = FactorAnalysis(n_components=25).fit(df_iv)

_.

partial = PLSRegression(n_components=10).fit_transform(df_iv, df_reg[[
        "granular_residual", 
        "network_residual",
    ]])[0]





df_iv = df_future["capm_future_error"][df_reg.index].unstack()
# df_iv = df_iv.fillna(df_iv.mean())
# df_iv = df_iv.fillna(df_iv.mean()).values
df_iv = df_iv.fillna(0)
# df_iv = PCA(n_components=10).fit_transform(df_iv)
# df_iv = SparsePCA(n_components=10).fit_transform(df_iv)
# df_iv = PLSRegression(n_components=10).fit_transform(df_iv, df_reg[[
#         "granular_residual", 
#         "network_residual",
#     ]])[0]

reg = IV2SLS(
    endog=df_reg[["ret_vw"]],
    exog=sm.tools.tools.add_constant(df_reg[[
        "granular_residual", 
        "network_residual",
    ]]).shift(0).fillna(0),
    instrument=sm.tools.add_constant(factors[:,:10]),#df_iv,
).fit()#cov_type="HAC",cov_kwds={'maxlags': 5})
reg.summary()

FullIV(
    endog=df_reg[["ret_vw"]],
    exog=sm.tools.tools.add_constant(df_reg[[
        "granular_residual", 
        "network_residual",
    ]]).shift(0).fillna(0),
    instruments=pd.DataFrame(sm.tools.add_constant(factors[:,:])),
).fit()._create_summary_column(add_outputs=["R-squared", "N"])

FullIV(
    endog=df_reg[["ret_vw"]],
    exog=sm.tools.tools.add_constant(df_reg[[
        "granular_residual", 
        # "network_residual",
    ]]).shift(0).fillna(0),
    instruments=pd.DataFrame(sm.tools.add_constant(factors[:,:])),
).fit()._create_summary_column(add_outputs=["R-squared", "N"])

FullIV(
    endog=df_reg[["ret_vw"]],
    exog=sm.tools.tools.add_constant(df_reg[[
        # "granular_residual", 
        "network_residual",
    ]]).shift(0).fillna(0),
    instruments=pd.DataFrame(sm.tools.add_constant(factors[:,:])),
).fit()._create_summary_column(add_outputs=["R-squared", "N"])





df_estimates.capm_mktrf.hist(bins=100)

df_estimates[["capm_mktrf"]].join(
    construct_residual_weights(df_estimates, variable="wfevd_out_connectedness", benchmark_variable="equal", constant_leverage=False).rename("equal")
).join(
    construct_residual_weights(df_estimates, variable="wfevd_out_connectedness", benchmark_variable="capm_mktrf", constant_leverage=False).rename("beta")
).corr()

df_estimates[["c4_mktrf", "c4_smb", "c4_hml", "c4_umd"]].join(
    construct_residual_weights(df_estimates, variable="mean_mcap", benchmark_variable="equal", constant_leverage=False).rename("granular")
).join(
    construct_residual_weights(df_estimates, variable="wfevd_out_connectedness", benchmark_variable="equal", constant_leverage=False).rename("network")
).corr()





# +
# df_reg=df_analysis[SPLIT_DATE:][["ret_vw"]+["granular_residual", "network_residual"]].dropna()
# df_iv = df_future["retadj"][df_reg.index].unstack().fillna(0)
# -

reg_table = kf.RegressionTable()

endog = "ret_vw"
exog = [
    "granular_residual",
    "network_residual",
]
df_reg=df_analysis[SPLIT_DATE:][[endog]+exog].dropna()
df_iv = df_future["capm_resid"][df_reg.index].unstack().fillna(0)
reg = sm.sandbox.regression.gmm.IV2SLS(endog=df_reg[endog],
                                       exog=sm.tools.tools.add_constant(df_reg[exog]).shift(0).fillna(0),
                                       instrument=df_iv,
                                      ).fit()#cov_type="HAC",cov_kwds={'maxlags': 5})
reg_table = reg_table.join_regression(reg, add_outputs=["R-squared", "N"])

reg_table









weights = construct_residual_weights(df_estimates, variable="mean_mcap", benchmark_variable="equal", constant_leverage=False)
weights *= df_estimates["capm_mktrf"]
index_beta = weights.groupby("sampling_date").sum()[SPLIT_DATE:]

index_beta.plot()

index_beta.mean()

weights = construct_residual_weights(df_estimates, variable="wfevd_out_connectedness", benchmark_variable="equal", constant_leverage=False)
weights *= df_estimates["capm_mktrf"]
index_beta = weights.groupby("sampling_date").sum()[SPLIT_DATE:]

index_beta.plot()

index_beta.mean()

weights = construct_residual_weights(df_estimates, variable="wfevd_out_connectedness", benchmark_variable="equal", constant_leverage=False)
weights *= df_estimates["capm_mktrf_next1M"]
index_beta = weights.groupby("sampling_date").sum()[SPLIT_DATE:]

index_beta.plot()

index_beta.mean()


