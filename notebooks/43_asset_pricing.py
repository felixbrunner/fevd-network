# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from euraculus.download import download_famafrench_dataset
from euraculus.factor import FactorModel

industry_portfolios_49 = download_famafrench_dataset(name="49_Industry_Portfolios_daily", key=0)

industry_portfolios_12 = download_famafrench_dataset(name="12_Industry_Portfolios_daily", key=0)

size_bm_portfolios = download_famafrench_dataset(name="25_Portfolios_5x5_Daily", key=0)

profit_invest_portfolios = download_famafrench_dataset(name="25_Portfolios_OP_INV_5x5_daily", key=0)

size_mom_portfolios = download_famafrench_dataset(name="25_Portfolios_ME_Prior_12_2_Daily", key=0)

industry_portfolios_12.cumsum().plot()



factor_data = download_famafrench_dataset(name="F-F_Research_Data_Factors_daily", key=0)



ff3f = FactorModel(factor_data=factor_data[["Mkt-RF", "SMB", "HML"]])

ff3f.fit(size_mom_portfolios)
ff3f.perform_grs_test()

ax = ff3f.plot_predictions()

ax = ff3f.plot_results()


