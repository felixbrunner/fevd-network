# %% [markdown]
# # 02 - Factor models estimation & decomposition
# This notebook contains:
# - monthly factor model estimation for:
#     - returns
#     - log variance
#
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
import numpy as np
import euraculus
# import kungfu as kf
import datetime as dt
from dateutil.relativedelta import relativedelta
from euraculus.data import DataMap
from euraculus.factor import FactorModel, SPY1FactorModel, CAPM, FamaFrench3FactorModel, Carhart4FactorModel, SPYVariance1FactorModel

# %% [markdown]
# ## Set up
# ### Data

# %%
data = DataMap("../data")
df_rf = data.load_rf()

# %% [markdown]
# ### Models

# %%
ret_models = {
    "spy_capm": SPY1FactorModel(data),
    "capm": CAPM(data),
    "ff3": FamaFrench3FactorModel(data),
    "c4": Carhart4FactorModel(data),
}

# %%
var_models = {
    "logvar_capm": SPYVariance1FactorModel(data),
}

# %% [markdown]
# ### Dates

# %%
# define timeframe
first_sampling_date = dt.datetime(year=1994, month=1, day=31)
last_sampling_date = dt.datetime(year=2021, month=12, day=31)

# %% [markdown]
# ## Standard Factor Models

# %% [markdown]
# ### Backward part

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get excess return samples
    df_back = data.load_sample(sampling_date.year, sampling_date.month, which="back", column="retadj")
    df_back -= df_rf.loc[df_back.index].values
    
    # estimate models backwards
    df_estimates, df_residuals = euraculus.factor.estimate_models(ret_models, df_back)
    
    # store
    data.store(data=df_residuals, path="samples/{0}{1:0=2d}/df_back.csv".format(sampling_date.year, sampling_date.month))
    data.store(data=df_estimates, path="samples/{0}{1:0=2d}/df_estimates.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed factor model estimation at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)

# %% [markdown]
# ### Forward part as expanding window

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get excess return samples
    df_forward = data.load_sample(sampling_date.year, sampling_date.month, which="forward", column="retadj")
    df_forward -= df_rf.loc[df_forward.index].values
    
    # slice expanding window
    df_expanding_estimates = pd.DataFrame(index=df_forward.columns)
    for window_length in range(1, 13):
        end_date = sampling_date + relativedelta(months=window_length, day=31)
        df_window = df_forward[df_forward.index <= end_date]
    
        # estimate models in window
        df_estimates, df_residuals = euraculus.factor.estimate_models(ret_models, df_window)
    
        # collect
        df_estimates = df_estimates.add_suffix("_next{}M".format(window_length))
        df_expanding_estimates = df_expanding_estimates.join(df_estimates)
    
    # store
    data.store(data=df_expanding_estimates, path="samples/{0}{1:0=2d}/df_estimates.csv".format(sampling_date.year, sampling_date.month))  
    data.store(data=df_residuals, path="samples/{0}{1:0=2d}/df_forward.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed factor model estimation at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)

# %% [markdown]
# ## Variance Factor Models

# %% [markdown]
# ### Backward part

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get excess return samples
    df_back = data.load_sample(sampling_date.year, sampling_date.month, which="back", column="var")
    df_back = np.log(df_back)
    
    # estimate models backwards
    df_estimates, df_residuals = euraculus.factor.estimate_models(var_models, df_back)
    
    # store
    data.store(data=df_residuals, path="samples/{0}{1:0=2d}/df_back.csv".format(sampling_date.year, sampling_date.month))
    data.store(data=df_estimates, path="samples/{0}{1:0=2d}/df_estimates.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed factor model estimation at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)

# %% [markdown]
# ### Forward part as expanding window

# %%
# %%time
sampling_date = first_sampling_date
while sampling_date <= last_sampling_date:
    # get excess return samples
    df_forward = data.load_sample(sampling_date.year, sampling_date.month, which="forward", column="var")
    df_forward = np.log(df_forward)
    
    # slice expanding window
    df_expanding_estimates = pd.DataFrame(index=df_forward.columns)
    for window_length in range(1, 13):
        end_date = sampling_date + relativedelta(months=window_length, day=31)
        df_window = df_forward[df_forward.index <= end_date]
    
        # estimate models in window
        df_estimates, df_residuals = euraculus.factor.estimate_models(var_models, df_window)
    
        # collect
        df_estimates = df_estimates.add_suffix("_next{}M".format(window_length))
        df_expanding_estimates = df_expanding_estimates.join(df_estimates)
    
    # store
    data.store(data=df_expanding_estimates, path="samples/{0}{1:0=2d}/df_estimates.csv".format(sampling_date.year, sampling_date.month))  
    data.store(data=df_residuals, path="samples/{0}{1:0=2d}/df_forward.csv".format(sampling_date.year, sampling_date.month))
    
    # increment monthly end of month
    print("Completed factor model estimation at {}".format(sampling_date.date()))
    sampling_date += relativedelta(months=1, day=31)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # OLD

# %%
# %%time
df_estimates = pd.DataFrame()
for year in range(1994, 1995):
    for month in range(1, 2):
        ## backwards part
        df_back = euraculus.loader.load_monthly_crsp(
            year, month, which="back", column="retadj"
        )
        df_back -= euraculus.loader.load_rf(year=year, month=month, which="back").values
        sampling_date = df_back.index[-1]
        n_months = len(df_back.index.to_period("M").unique())

        # estimate
        back_estimates, residuals = euraculus.factor.estimate_models(
            df_back, ret_models, return_residuals=True
        )
        back_estimates = back_estimates.merge(
            residuals.unstack().var().unstack(level=0).add_suffix("_sigma2"),
            left_index=True,
            right_on="permno",
        )
        back_estimates.index = pd.MultiIndex.from_product(
            [[sampling_date], back_estimates.index], names=["sampling_date", "permno"]
        )
        back_estimates.columns = [
            column + "_back_{}M".format(n_months) for column in back_estimates.columns
        ]

        ## forward part
        df_forward = euraculus.loader.load_monthly_crsp(
            year, month, which="forward", column="retadj"
        )
        df_forward -= euraculus.loader.load_rf(
            year=year, month=month, which="forward"
        ).values
        months = df_forward.index.to_period("M").unique().tolist()

        # estimate
        forward_estimates = pd.DataFrame(index=back_estimates.index)
        for i in range(1, len(months) + 1):
            data = df_forward[df_forward.index.to_period("M").isin(months[0:i])]
            month_estimates, residuals = euraculus.factor.estimate_models(
                data, ret_models, return_residuals=True
            )
            month_estimates = month_estimates.merge(
                residuals.unstack().var().unstack(level=0).add_suffix("_sigma2"),
                left_index=True,
                right_on="permno",
            )
            month_estimates.index = pd.MultiIndex.from_product(
                [[sampling_date], month_estimates.index],
                names=["sampling_date", "permno"],
            )
            month_estimates.columns = [
                column + "_forward_{}M".format(i) for column in month_estimates.columns
            ]
            forward_estimates = forward_estimates.join(month_estimates)

        # combine
        estimates = back_estimates.join(forward_estimates)
        df_estimates = df_estimates.append(estimates)
    print("Done estimating year {}".format(year))

#     df_estimates.to_csv("../data/estimated/factor_exposures.csv")

# %%


# %%


# %%


# %%
# # %%time
# df_estimates = pd.DataFrame()
# for year in range(1994, 2021):
#     for month in range(1, 12+1):
#         ## backwards part
#         df_back = euraculus.loader.load_monthly_crsp(year, month, which='back', column='retadj')
#         df_back -= euraculus.loader.load_rf(year=year, month=month, which='back').values
#         sampling_date = df_back.index[-1]
#         n_months = len(df_back.index.to_period('M').unique())

#         # estimate
#         back_estimates = euraculus.factor.estimate_models(df_back, ret_models)
#         back_estimates.index = pd.MultiIndex.from_product([[sampling_date], back_estimates.index], names=['sampling_date', 'permno'])
#         back_estimates.columns = [column+'_back_{}M'.format(n_months) for column in back_estimates.columns]

#         ## forward part
#         df_forward = euraculus.loader.load_monthly_crsp(year, month, which='forward', column='retadj')
#         df_forward -= euraculus.loader.load_rf(year=year, month=month, which='forward').values
#         months = df_forward.index.to_period('M').unique().tolist()

#         # estimate
#         forward_estimates = pd.DataFrame(index=back_estimates.index)
#         for i in range(1, len(months)+1):
#             data = df_forward[df_forward.index.to_period('M').isin(months[0:i])]
#             month_estimates = euraculus.factor.estimate_models(data, ret_models)
#             month_estimates.index = pd.MultiIndex.from_product([[sampling_date], month_estimates.index], names=['sampling_date', 'permno'])
#             month_estimates.columns = [column+'_forward_{}M'.format(i) for column in month_estimates.columns]
#             forward_estimates = forward_estimates.join(month_estimates)

#         # combine
#         estimates = back_estimates.join(forward_estimates)
#         df_estimates = df_estimates.append(estimates)
#     print('Done estimating year {}'.format(year))

#     df_estimates.to_csv('../data/estimated/factor_exposures.csv')

# %% [markdown]
# ## Construct CAPM idiosyncratic variances

# %%
# load betas
betas = pd.read_csv("../data/estimated/factor_exposures.csv")
betas["sampling_date"] = pd.to_datetime(betas["sampling_date"])
betas = betas.set_index(["sampling_date", "permno"])["spy_capm_spy_back_12M"]

# load SPY data
spy_var = euraculus.loader.load_spy(columns="var")

# %%
# %%time
for year in range(1994, 2021):
    for month in range(1, 12 + 1):
        # period inputs
        total_variance = euraculus.loader.load_monthly_crsp(
            year, month, which="back", column="var"
        )
        period_betas = betas[
            (betas.index.get_level_values("sampling_date").year == year)
            & (betas.index.get_level_values("sampling_date").month == month)
        ]
        period_spy_var = spy_var[spy_var.index.isin(total_variance.index)]

        # decompose
        df_decomposition = euraculus.factor.decompose_variance(
            total_variance, period_betas, period_spy_var
        )
        df_decomposition.to_csv(
            "../data/processed/monthly/{}/{}/df_var_decomposition.csv".format(
                year, month
            )
        )

    print("Done decomposiong year {}".format(year))

# %% [markdown]
# ## Variance Series Factor Models
# ### Set up factor models

# %%
logvar_capm = kf.FactorModel(np.log(df_spy["var"]).rename("spy"))
var_models = {"logvar_capm": logvar_capm}

# %% [markdown]
# ### Estimate

# %%
# %%time
df_estimates = pd.DataFrame()
for year in range(1994, 2021):
    for month in range(1, 12 + 1):
        ## backwards part
        df_back = euraculus.loader.load_monthly_crsp(
            year, month, which="back", column="var"
        )
        df_back = np.log(df_back.replace(0, value=df_back.mask(df_back <= 0).min()))
        sampling_date = df_back.index[-1]
        n_months = len(df_back.index.to_period("M").unique())

        # estimate
        back_estimates, back_residuals = euraculus.factor.estimate_models(
            df_back, var_models, return_residuals=True
        )
        back_residuals.to_csv(
            "../data/processed/monthly/{}/{}/df_back_residuals.csv".format(year, month)
        )
        back_estimates.index = pd.MultiIndex.from_product(
            [[sampling_date], back_estimates.index], names=["sampling_date", "permno"]
        )
        back_estimates.columns = [
            column + "_back_{}M".format(n_months) for column in back_estimates.columns
        ]

        ## forward part
        df_forward = euraculus.loader.load_monthly_crsp(
            year, month, which="forward", column="var"
        )
        df_forward = np.log(
            df_forward.replace(0, value=df_forward.mask(df_forward <= 0).min())
        )
        months = df_forward.index.to_period("M").unique().tolist()

        # estimate
        forward_estimates = pd.DataFrame(index=back_estimates.index)
        for i in range(1, len(months) + 1):
            data = df_forward[df_forward.index.to_period("M").isin(months[0:i])]
            month_estimates = euraculus.factor.estimate_models(data, var_models)
            month_estimates.index = pd.MultiIndex.from_product(
                [[sampling_date], month_estimates.index],
                names=["sampling_date", "permno"],
            )
            month_estimates.columns = [
                column + "_forward_{}M".format(i) for column in month_estimates.columns
            ]
            forward_estimates = forward_estimates.join(month_estimates)

        # combine
        estimates = back_estimates.join(forward_estimates)
        df_estimates = df_estimates.append(estimates)
    print("Done estimating year {}".format(year))

    df_estimates.to_csv("../data/estimated/logvar_factor_exposures.csv")

# %% [markdown]
# ## Analyse
# ### Beta Estimates

# %%
all_models = pd.read_csv("../data/estimated/factor_exposures.csv")
all_models["sampling_date"] = pd.to_datetime(all_models.sampling_date)

# %%
df_analysis = df_estimates.merge(
    all_models, left_index=True, right_on=["sampling_date", "permno"], how="outer"
)[
    [
        "logvar_capm_alpha_back_12M",
        "logvar_capm_spy_back_12M",
        "capm_alpha_back_12M",
        "capm_mktrf_back_12M",
    ]
]

# %%
df_analysis["squared_betas"] = df_analysis["logvar_capm_spy_back_12M"] ** 2

# %%
df_analysis.corr()

# %% [markdown]
# ### Idiosyncratic Variance Residuals

# %%
year = 2011
month = 2

# %%
df_back_residuals = pd.read_csv(
    "../data/processed/monthly/{}/{}/df_back_residuals.csv".format(year, month)
)
df_var_decomposition = pd.read_csv(
    "../data/processed/monthly/{}/{}/df_var_decomposition.csv".format(year, month)
)

# %%
df_back_residuals["logvar_capm"].corr(np.log(df_var_decomposition["idiosyncratic"]))

# %%
correlations = pd.Series(index=pd.MultiIndex.from_product([[], []]))
for year in range(1994, 2020):
    for month in range(1, 12 + 1):
        # load
        df_back_residuals = pd.read_csv(
            "../data/processed/monthly/{}/{}/df_back_residuals.csv".format(year, month)
        )
        df_var_decomposition = pd.read_csv(
            "../data/processed/monthly/{}/{}/df_var_decomposition.csv".format(
                year, month
            )
        )

        # calculate
        correlation = df_back_residuals["logvar_capm"].corr(
            np.log(df_var_decomposition["idiosyncratic"])
        )

        # collect
        correlations[(year, month)] = correlation

# %%
correlations.plot()

# %% [markdown]
# ## Index correlation

# %%
df_indices = df_spy[["ret"]].join(df_factors["rf"]).join(df_factors["mktrf"])
df_indices["spy_excess"] = df_indices["ret"] - df_indices["rf"]
df_indices[["spy_excess", "mktrf"]].corr()

# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Single year analysis
# ### Load data

# %%
year = 2019

# %%
df_returns = euraculus.loader.load_year(year, data="returns")
df_volas = euraculus.loader.load_year(year, data="volas")
df_spy_ret = euraculus.loader.load_spy(year, columns=['ret'])
df_spy_vola = euraculus.loader.load_spy(year, columns=['var'])**0.5
rf = euraculus.loader.load_rf(year)
tickers = euraculus.loader.load_year_tickers(year)

# %%
returns, factors = df_returns, df_spy_ret
estimates = euraculus.factor.estimate_factor_model(
    returns.subtract(rf.values), factors.subtract(rf.values)
)

volas, betas, factor_vola = df_volas, estimates["beta"], df_spy_vola
euraculus.preprocess.decompose_vola(volas, betas, factor_vola)

# %% [markdown]
# ### Compare zero innovation frequency for CAPM regression vs. Variance regression

# %%
def idiosyncratic_vola(year, steps=2):
    """"""
    # setup
    index = euraculus.loader.load_year(year, data="returns").columns
    df_estimates = pd.DataFrame(index=index)

    for step in range(steps):
        # load
        df_returns = euraculus.loader.load_year(
            year, data="returns", data_year=year + step
        )
        df_volas = euraculus.loader.load_year(year, data="volas", data_year=year + step)
        df_spy_ret = euraculus.loader.load_spy_returns(year + step)
        df_spy_vola = euraculus.loader.load_spy_vola(year + step)

        # estimate
        estimates = estimate_factor_model(df_returns, df_spy_ret)
        vola_estimates = estimate_factor_model(df_volas ** 2, df_spy_vola ** 2)

        # decompose
        idio_vola = decompose_vola(df_volas, estimates["beta"], df_spy_vola)
        idio_vola_direct = decompose_vola(
            df_volas, estimates["beta"] ** 0.5, df_spy_vola
        )

    return idio_vola, idio_vola_direct


# %%
zero_count = pd.DataFrame(
    index=np.arange(1994, 2020), columns=["CAPM regression", "Variance regression"]
)
for year in range(1994, 2020):
    CAPM, VOLA = idiosyncratic_vola(year, 1)
    zeros_CAPM = (CAPM == 0).values.sum()
    zeros_VOLA = (VOLA == 0).values.sum()
    zero_count.loc[year, "CAPM regression"] = zeros_CAPM
    zero_count.loc[year, "Variance regression"] = zeros_VOLA

# %%
(zero_count / (250 * 100)).plot()

# %%
estimate_factor_model(df_volas, df_spy_vola).join(
    estimate_factor_model(df_returns, df_spy_ret), lsuffix="vola"
).corr()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## OLD: Process all years

# %%
# %%time
df_factor_model = pd.DataFrame()
for year in range(1994, 2020):
    # estimate
    if year == 2019:
        df_estimates = euraculus.factor.idiosyncratic_vola(year, steps=1)
    else:
        df_estimates = euraculus.factor.idiosyncratic_vola(year, steps=2)
    df_estimates.index = pd.MultiIndex.from_product(
        [[year], df_estimates.index], names=["sampling_year", "permno"]
    )

    # collect
    df_factor_model = df_factor_model.append(df_estimates)

    # progress
    print("processed sampling year {}".format(year))

df_factor_model.to_csv("../data/estimated/factor_models.csv")

# %% [markdown]
# ## Statistical Factor Analysis

# %%
from sklearn.decomposition import FactorAnalysis

# %%
def build_factor_model_residuals(year):
    # load
    df_returns = euraculus.loader.load_year(
        year, data="returns", data_year=year
    ).fillna(0)
    df_vola = euraculus.loader.load_year(year, data="volas", data_year=year)
    df_idio = euraculus.loader.load_year(year, data="idio_vola", data_year=year)
    rf = euraculus.loader.load_rf(year)
    ff = euraculus.loader.load_factors(year)

    # estimate factor models
    mm_estimates = euraculus.factor.estimate_factor_model(
        df_returns.subtract(rf.values), ff[["mktrf"]]
    )
    ff3_estimates = euraculus.factor.estimate_factor_model(
        df_returns.subtract(rf.values), ff[["mktrf", "smb", "hml"]]
    )
    c4_estimates = euraculus.factor.estimate_factor_model(
        df_returns.subtract(rf.values), ff[["mktrf", "smb", "hml", "umd"]]
    )

    fam = FactorAnalysis(n_components=5).fit(df_returns.T)
    facs = pd.DataFrame(fam.components_).T
    fam_estimates = euraculus.factor.estimate_factor_model(
        df_returns.subtract(rf.values), facs
    )

    # construct residuals
    mm_residuals = (
        df_returns
        - pd.DataFrame(index=ff.index, data={"alpha": 1}).join(ff[["mktrf"]])
        @ mm_estimates.T
    )
    ff3_residuals = (
        df_returns
        - pd.DataFrame(index=ff.index, data={"alpha": 1}).join(
            ff[["mktrf", "smb", "hml"]]
        )
        @ ff3_estimates.T
    )
    c4_residuals = (
        df_returns
        - pd.DataFrame(index=ff.index, data={"alpha": 1}).join(
            ff[["mktrf", "smb", "hml", "umd"]]
        )
        @ c4_estimates.T
    )
    fam_residuals = (
        df_returns
        - pd.DataFrame(index=ff.index, data={"alpha": 1}).join(
            pd.DataFrame(index=ff.index, data=facs.values)
        )
        @ fam_estimates.T
    )

    # save matrices
    mm_residuals.to_csv(
        "../data/processed/annual/{}/mm_residuals_{}.csv".format(year, year)
    )
    ff3_residuals.to_csv(
        "../data/processed/annual/{}/ff3_residuals_{}.csv".format(year, year)
    )
    c4_residuals.to_csv(
        "../data/processed/annual/{}/c4_residuals_{}.csv".format(year, year)
    )
    fam_residuals.to_csv(
        "../data/processed/annual/{}/fam_residuals_{}.csv".format(year, year)
    )


# %%
for year in range(1994, 2020):
    build_factor_model_residuals(year)
    print("Done building year {}".format(year))

# %%

_ = FactorAnalysis(n_components=5).fit(df_returns.T)
facs = pd.DataFrame(_.components_).T

# %%
year = 2000
df_returns = euraculus.loader.load_year(year, data="returns", data_year=year)
df_vola = euraculus.loader.load_year(year, data="volas", data_year=year)
df_idio = euraculus.loader.load_year(year, data="idio_vola", data_year=year)
rf = euraculus.loader.load_rf(year)
ff = euraculus.loader.load_factors(year)
estimates = euraculus.factor.estimate_factor_model(df_returns.subtract(rf.values), facs)
