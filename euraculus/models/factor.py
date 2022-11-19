"""
This module provides functions to carry out factor analysis in general,
and on CRSP data in particular.

"""

import numpy as np
import pandas as pd

from euraculus.data.map import DataMap
from kungfu import FactorModel


class SPY1FactorModel(FactorModel):
    """1-Factor model with SPY returns as a single factor."""

    def __init__(self, datamap: DataMap):
        """Load and store SPY factor data."""
        df_spy = datamap.load_spy_data()["ret"].rename("spy")
        super().__init__(factor_data=df_spy)


class CAPM(FactorModel):
    """CAPM 1-factor model."""

    def __init__(self, datamap: DataMap):
        """Load and store market factor data."""
        df_capm = datamap.load_famafrench_factors(model="capm")
        self.factor_data = df_capm


class FamaFrench3FactorModel(FactorModel):
    """Fama/French 3-Factor model with market, value, and size factors."""

    def __init__(self, datamap: DataMap):
        """Load and store mktrf, hml, smb factor data."""
        df_ff3 = datamap.load_famafrench_factors(model="ff3f")
        self.factor_data = df_ff3


class Carhart4FactorModel(FactorModel):
    """Carhart 4-Factor model with market, value, size, and momentum factors."""

    def __init__(self, datamap: DataMap):
        """Load and store mktrf, hml, smb, umd factor data."""
        df_c4 = datamap.load_famafrench_factors(model="c4f")
        self.factor_data = df_c4


class SPYVariance1FactorModel(FactorModel):
    """Variance 1-Factor model with intraday log variance of SPY as single factor."""

    def __init__(self, datamap: DataMap):
        """Load and store mktrf, hml, smb, umd factor data."""
        df_spy_var = np.log(datamap.load_spy_data(series="var")).rename(
            columns={"var": "spylogvar"}
        )
        self.factor_data = df_spy_var


def estimate_models(models: dict, data: pd.DataFrame) -> tuple:
    """Estimate a set of models and collect the results.

    Args:
        models: Name, FactorModel pairs to define the set of models estimated.
        data: The returns data to estimate the factor models on.

    Returns:
        df_estimates: Point estimates for all estimated models.
        df_residuals: Model residuals (pricing errors) for all estimated models.

    """
    # set up storage arrays
    df_estimates = pd.DataFrame(index=data.columns)
    df_residuals = pd.DataFrame(index=data.stack().index)

    # estimate
    for name, model in models.items():
        model.fit(data)

        # collect results
        df_estimates = df_estimates.join(
            model.coef_.T.add_prefix("{}_".format(name))
        )  # coefficients
        df_estimates = df_estimates.join(
            model.sigma2_.to_frame().add_prefix("{}_".format(name))
        )  # idio var
        df_estimates = df_estimates.join(
            model.r2_.to_frame().add_prefix("{}_".format(name))
        )  # r2
        df_residuals[name + "_resid"] = model.residuals_.stack()  # resiudals

    return (df_estimates, df_residuals)


def decompose_variance(
    asset_variances: pd.DataFrame,
    factor_betas: pd.DataFrame,
    factor_variances: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose intraday asset variances into systematic and non-systematic parts.

    Args:
        asset_variances: High-frequency asset variance observations (T x N).
        factor_betas: Factor exposures of each asset (N x 1).
        factor_variances: High-frequency factor variance observations (T x 1).

    Returns:
        df_decomposition: Decomposition of total intraday variances into:
            var_systematic = var_factor * beta^2
            var_idiosyncratic = max(var_total - var_systematic, 0)

    """
    # systematic variance
    if isinstance(factor_variances, pd.Series):
        factor_variances = factor_variances.to_frame()
    k_factors = factor_variances.shape[1]
    systematic_var = factor_variances @ factor_betas.values.reshape(k_factors, -1) ** 2
    systematic_var.columns = asset_variances.columns

    # idiosyncratic variance
    idio_var = asset_variances - systematic_var
    idio_var[idio_var <= 0] = np.nan

    # output
    df_decomposition = asset_variances.stack(dropna=False).rename("tot").to_frame()
    df_decomposition["sys"] = systematic_var.stack(dropna=False).values
    df_decomposition["idio"] = idio_var.stack(dropna=False).values

    return df_decomposition
