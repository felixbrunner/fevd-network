"""
This module provides functions to carry out factor analysis in general,
and on CRSP data in particular.

"""
import numpy as np
import pandas as pd
import warnings
from euraculus.data import DataMap


class FactorModel:
    """Financial linear factor model.

    Attributes:
        factor_data: The factor data index by DatetimeIndex.
        is_fitted: Indicates if model is fitted to asset returns data.

    """

    def __init__(self, factor_data: pd.DataFrame):
        """Store factor data to be used in the model."""
        self.factor_data = factor_data
        self.is_fitted = False

    @property
    def factor_data(self) -> pd.DataFrame:
        """The factor returns data used in the model as a dataframe."""
        return self._factor_data

    @factor_data.setter
    def factor_data(self, factor_data: pd.DataFrame):

        # check if data is indexed by datetime
        if not type(factor_data.index) == pd.DatetimeIndex:
            raise ValueError(
                "factor_data needs to have a DatetimeIndex, index has type '{}'".format(
                    type(factor_data.index)
                )
            )

        # transform to dataframe if series
        if isinstance(factor_data, pd.Series):
            factor_data = factor_data.to_frame()

        # set attribute
        self._factor_data = factor_data

    @property
    def k_factors(self) -> int:
        """The number of factors in the factor model."""
        return self.factor_data.shape[1]

    @staticmethod
    def _preprocess_returns_data(returns_data: pd.DataFrame) -> pd.DataFrame:
        """Set up returns timeseries data as a DataFrame in wide format.

        Args:
            returns_data: The asset returns data in any DataFrame format.

        Returns:
            returns_data: The processed returns data in a T by N DataFrame.

        """

        # unstack multiindex
        if type(returns_data.index) == pd.MultiIndex:
            if len(returns_data.columns) != 1:
                raise ValueError("too many columns, supply only return data")
            returns_data = returns_data.unstack()

        # check if returns data is indexed by datetime
        if not type(returns_data.index) == pd.DatetimeIndex:
            raise ValueError(
                "returns_data needs to have a DatetimeIndex, index has type '{}'".format(
                    type(returns_data.index)
                )
            )

        # transform to dataframe if series
        if isinstance(returns_data, pd.Series):
            returns_data = returns_data.to_frame()

        return returns_data

    def _preprocess_factor_data(
        self, returns_data: pd.DataFrame, add_constant: bool
    ) -> pd.DataFrame:
        """Set up factor data to match asset returns data index.

        Args:
            returns_data: The asset returns data in any DataFrame format.
            add_constant: Indicates if constant should be included.

        Returns:
            factor_data: Readily processed factor data in a T by K DataFrame.

        """
        # set up index and constant
        factor_data = pd.DataFrame(index=returns_data.index)
        if add_constant:
            factor_data["const"] = 1

        # fill in factor data
        factor_data = factor_data.merge(
            self.factor_data,
            how="left",
            left_index=True,
            right_index=True,
        )

        # warn if factor data is missing
        if factor_data.isna().sum().sum() > 0:
            warnings.warn(
                "filling in missing factor observations (out of {}) with zeros: \n{}".format(
                    len(factor_data), factor_data.isna().sum()
                )
            )
            factor_data = factor_data.fillna(0)

        return factor_data

    def _set_up_attributes(self, returns: pd.DataFrame, factors: pd.DataFrame):
        """Set up storage arrays for fitting results.

        Args:
            returns: The preprocessed asset returns data.
            factors: The preprocessed factor data.

        """
        # K(+1) times N attributes
        self._coef_ = pd.DataFrame(index=factors.columns, columns=returns.columns)
        self._se_ = pd.DataFrame(index=factors.columns, columns=returns.columns)

        # N times 1 attributes
        self._sigma2_ = pd.Series(index=returns.columns, name="sigma2")
        self._r2_ = pd.Series(index=returns.columns, name="R2")

        # T times N attributes
        self._fitted_ = pd.DataFrame(index=returns.index, columns=returns.columns)
        self._resid_ = pd.DataFrame(index=returns.index, columns=returns.columns)

    @staticmethod
    def _regress(returns_data: pd.Series, factor_data: pd.DataFrame) -> dict:
        """Calculate factor model regression for a single asset.

        Method will calculate regression coefficients and other statistics and
        return a dictionary with the results.

        Args:
            returns_data: The preprocessed asset returns data.
            factor_data: The preprocessed factor data.

        Returns:
            regression_stats: The regression results.

        """
        # set up
        observations = returns_data.notna()
        X = factor_data.loc[observations].values
        y = returns_data[observations].values

        # calculate
        if observations.sum() >= X.shape[1]:
            coef = np.linalg.inv(X.T @ X) @ (X.T @ y)
        else:
            coef = np.full(
                shape=[
                    X.shape[1],
                ],
                fill_value=np.nan,
            )
            warnings.warn(
                "not enough observations to estimate factor loadings for {}".format(
                    returns_data.name
                )
            )
        fitted = X @ coef
        resid = y - fitted
        sigma2 = (resid ** 2).sum() / (len(y) - X.shape[1])
        if observations.sum() >= X.shape[1]:
            se = sigma2 * np.diag(np.linalg.inv(X.T @ X))
        else:
            se = np.full(
                shape=[
                    X.shape[1],
                ],
                fill_value=np.nan,
            )
        r2 = 1 - sigma2 / y.var()

        # collect
        regression_stats = {
            "name": returns_data.name,
            "coef": coef,
            "fitted": fitted,
            "resid": resid,
            "se": se,
            "sigma2": sigma2,
            "r2": r2,
            "index": returns_data.index[observations],
        }
        return regression_stats

    def _store_regression_stats(self, stats: dict):
        """Store the results of a factor regression in the storage arrays.

        Args:
            stats: Factor regression results.

        """
        self._coef_.loc[:, stats["name"]] = stats["coef"]

        # K(+1) times N attributes
        self._coef_.loc[:, stats["name"]] = stats["coef"]
        self._se_.loc[:, stats["name"]] = stats["se"]

        # N times 1 attributes
        self._sigma2_.loc[stats["name"]] = stats["sigma2"]
        self._r2_.loc[stats["name"]] = stats["r2"]

        # T times N attributes
        self._fitted_.loc[stats["index"], stats["name"]] = stats["fitted"]
        self._resid_.loc[stats["index"], stats["name"]] = stats["resid"]

    def fit(self, returns_data: pd.DataFrame, add_constant: bool = True):
        """Fit the factor model to an array of returns data.

        Args:
            returns_data: Asset returns data indexed by a DatetimeIndex.
            add_constant: Indicates if model is to be estimated with alpha.

        """
        # prepare
        returns_data = self._preprocess_returns_data(returns_data=returns_data)
        factor_data = self._preprocess_factor_data(
            returns_data=returns_data, add_constant=add_constant
        )
        self._set_up_attributes(returns=returns_data, factors=factor_data)

        # run regressions
        for asset, asset_returns in returns_data.items():
            regression_stats = self._regress(
                returns_data=asset_returns, factor_data=factor_data
            )
            self._store_regression_stats(stats=regression_stats)

        # update
        self.is_fitted = True

    @property
    def coef_(self):
        """The estimated model coefficients."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._coef_

    @property
    def alphas_(self):
        """The estimated model alphas."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        elif not "const" in self._coef_.index:
            raise AttributeError("model fitted without intercept")
        else:
            return self._coef_.loc["const"]

    @property
    def betas_(self):
        """The estimated factor loadings."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        elif "const" in self._coef_.index:
            return self._coef_.iloc[1:, :].T
        else:
            return self._coef_.T

    @property
    def se_(self):
        """The estimated coefficient standard errors."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._se_

    @property
    def sigma2_(self):
        """The estimated idiosyncratic volatilities."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._sigma2_

    @property
    def r2_(self):
        """The estimated idiosyncratic volatilities."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._r2_

    @property
    def fitted_(self):
        """The model fitted values."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._fitted_

    @property
    def residuals_(self):
        """The model residuals."""
        if not self.is_fitted:
            raise AttributeError("model is not fitted")
        else:
            return self._resid_


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
