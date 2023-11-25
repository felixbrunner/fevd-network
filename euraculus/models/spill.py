"""This module provides VAR and FactorVAR classes for estimation."""

import copy
import warnings

import pandas as pd
import numpy as np
import scipy as sp

import autograd.numpy as anp
import autograd.scipy as asp
from autograd import jacobian, hessian


class ReturnSpilloverModel:
    """Return spillover network model.

    Instances of this class allow to perform granular asset pricing analysis.

    Attributes:
        has_intercepts: Indicates if the VAR model includes a constant vector.
        spillover_proxy: Matrix with spillover proxies (off-diagonal).
        is_fitted: Indicates if the VAR model has been fitted to data.

    Additional attributes:
        intercepts_: The model intercepts.
        factor_loadings_: The model factor loadings.
        k_factors_: The number of factors in the model.
        scaling_constant_: Constant to define the magnitude of the network.
        spillover_matrix_: The fitted spillover proxy scaled to the data.
    """

    def __init__(
        self,
        network_proxy = np.ndarray,
        has_intercepts: bool = True,
        left_scaling: bool = False,
        right_scaling: bool = False,
    ):
        """Initiates the VAR object with descriptive attributes."""
        self.has_intercepts = has_intercepts
        self.spillover_proxy = network_proxy - np.diag(np.diag(network_proxy))
        self.left_scaling = left_scaling
        self.right_scaling = right_scaling
        self.is_fitted = False

    @property
    def n_series(self) -> int:
        """The number of variables in the network model."""
        return self.spillover_proxy.shape[1]
    
    @property
    def k_factors_(self) -> int:
        """The number of common factors in the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self._k_factors_

    @property
    def intercepts_(self) -> np.ndarray:
        """Vector of model intercepts."""
        if not self.has_intercepts:
            raise ValueError("Model does not have intercepts")
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        indices = np.arange(0, self.n_series * self.has_intercepts)
        intercepts_ = self.coef_[indices].reshape(self.n_series, 1)
        return intercepts_

    @property
    def factor_loadings_(self) -> np.ndarray:
        """Matrix with factor loadings."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        if self.k_factors_ > 0:
            indices = np.arange(self.n_series * self.has_intercepts, self.n_series * (self.has_intercepts + self.k_factors_))
            factor_loadings = self.coef_[indices].reshape(self.n_series, self.k_factors_)
        else:
            factor_loadings = np.ndarray([self.n_series, 0])
        return factor_loadings
    
    @property
    def scaling_coefficient_(self) -> float:
        """Scalar to scale the spillover proxy."""
        if self.left_scaling or self.right_scaling:
            raise ValueError("Model does not have scalar scaling.")
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self.coef_[-1]
    
    @property
    def scaling_constraint(self) -> float:
        """Upper bound for the scaling coefficient for a stable process."""
        return 1/abs(anp.linalg.eigvals(self.spillover_proxy)).max()

    
    @property
    def left_scaling_vector_(self) -> float:
        """Matrix with factor loadings."""
        if not self.left_scaling:
            raise ValueError("Model does no have left vector scaling.")
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        scaling_vector = self.coef_[
                self.n_series*(1+self.k_factors_):self.n_series*(2+self.k_factors_)
            ].reshape(self.n_series, 1)
        return scaling_vector
    
    @property
    def right_scaling_vector_(self) -> float:
        """Matrix with factor loadings."""
        if not self.right_scaling:
            raise ValueError("Model does no have right vector scaling.")
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        scaling_vector = self.coef_[-self.n_series:].reshape(self.n_series, 1)
        return scaling_vector
    
    @property
    def max_eigval(self) -> np.ndarray:
        """The largest eigenvalue of the spillover matrix in modulus."""
        return abs(np.linalg.eigvals(self.spillover_matrix_)).max()
    
    @property
    def spillover_matrix_(self) -> np.ndarray:
        """The fitted spillover proxy scaled to the data."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        if self.left_scaling:
            if self.right_scaling:
                return np.diag(self.left_scaling_vector_.squeeze()) @ self.spillover_proxy @ np.diag(self.right_scaling_vector_.squeeze())
            else:
                return np.diag(self.left_scaling_vector_.squeeze()) @ self.spillover_proxy
        else:
            if self.right_scaling:
                return self.spillover_proxy @ np.diag(self.right_scaling_vector_.squeeze())
            else:
                return self.scaling_coefficient_ * self.spillover_proxy
    
    @property
    def structural_matrix_(self) -> np.ndarray:
        """The structural matrix that describes the fitted returns.
        
        (I - cD)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return np.eye(self.n_series) - self.spillover_matrix_
    
    @property
    def inv_structural_matrix_(self) -> np.ndarray:
        """The structural matrix that describes the fitted returns.
        
        (I - cD)^-1
        """
        return np.linalg.inv(self.structural_matrix_)
    
    @property
    def adjusted_intercepts_(self) -> np.ndarray:
        """Intercepts adjusted for spillover effects."""
        return self.inv_structural_matrix_ @ self.intercepts_
    
    @property
    def adjusted_factor_loadings_(self) -> np.ndarray:
        """Factor loadings adjusted for spillover effects."""
        return self.inv_structural_matrix_ @ self.factor_loadings_

    def _build_y(self, return_data: np.ndarray) -> np.ndarray:
        """Create an array of dependent variables reshaped for estimation.

        Args:
            return_data: (t_periods, n_series) array with return observations.

        Returns:
            y: (t_periods*n_series,) array reshaped for regression form.
        """
        y = return_data.values.reshape(-1, 1, order="F")
        return y

    def _build_X(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
    ) -> sp.sparse.csc.csc_matrix:
        """Create a sparse diagonal block matrix of independent data for estimation.

        Args:
            return_data: (t_periods, n_series) array with return observations.
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form.
        """
        # size
        t_periods = return_data.shape[0]
        n_series = return_data.shape[1]
        elements = []

        # build
        if self.has_intercepts:
            elements += [sp.sparse.kron(sp.sparse.eye(n_series), np.ones([t_periods, 1]), format="csc")]
        elements += [sp.sparse.kron(sp.sparse.eye(n_series), factor_data, format="csc")]
        elements += [(self.spillover_proxy @ return_data.values.T).T.reshape(-1, 1, order="F")]

        X = sp.sparse.hstack(elements, format="csc")
        return X

    def _build_X_y(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
    ) -> tuple:
        """Create data matrices for estimation.

        Args:
            return_data: (t_periods, n_series) array with return observations.
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            y: (t_periods*n_series,) array reshaped for regression form.
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form.
        """
        X = self._build_X(
            return_data=return_data,
            factor_data=factor_data,
        )
        y = self._build_y(return_data=return_data)
        return (X, y)

    def fit_ols(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Fits the coefficients using OLS.

        Args:
            return_data: (t_periods, n_series) array with return observations.
            factor_data: (t_periods, k_factors) array with factor observations.
        """
        # build inputs
        assert self.n_series == return_data.shape[1], "data dimension does not match network"
        X, y = self._build_X_y(
            return_data=return_data,
            factor_data=factor_data,
        )

        # estimate
        coef_ = np.asarray(np.linalg.inv((X.T @ X).todense()) @ (X.T @ y))
        fitted_ = X @ coef_
        resid_ = y - fitted_
        sigma2_ = (resid_**2).sum() / (len(y) - X.shape[1])
        se_ = np.sqrt(sigma2_ * np.diag(np.linalg.inv((X.T @ X).todense())))

        # store coefficient estimates
        if hasattr(self, "information_matrix_"):
            del self.information_matrix_
        self.coef_ = coef_
        self.se_ = se_
        self._k_factors_ = factor_data.shape[1]
        self.is_fitted = True

    def _build_omega(
        self,
        return_data: np.ndarray,
    ) -> tuple:
        """Create weighting matrix for WLS estimation.

        Args:
            return_data: (t_periods, n_series) array with return observations.

        Returns:
            y: (t_periods*n_series,) array reshaped for regression form.
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form.
        """
        t_obs = return_data.shape[0]
        asset_weights = return_data.var()**-1
        importance_weights = np.kron(asset_weights, np.ones(t_obs))
        omega = sp.sparse.diags(importance_weights, 0, format="csc")
        return omega

    def fit_wls(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray = None,
    ) -> None:
        """Fits the coefficients using weighted least squares.

        Args:
            return_data: (t_periods, n_series) array with return observations.
            factor_data: (t_periods, k_factors) array with factor observations.
        """
        # build inputs
        assert self.n_series == return_data.shape[1], "data dimension does not match network"
        X, y = self._build_X_y(
            return_data=return_data,
            factor_data=factor_data,
        )
        omega = self._build_omega(return_data=return_data)

        # estimate
        coef_ = np.asarray(np.linalg.inv((X.T @ omega @ X).todense()) @ (X.T @ omega @ y))
        fitted_ = X @ coef_
        resid_ = y - fitted_
        sigma2_ = (resid_**2).sum() / (len(y) - X.shape[1])
        se_ = np.sqrt(sigma2_ * np.diag(np.linalg.inv((X.T @ X).todense())))

        # store coefficient estimates
        if hasattr(self, "information_matrix_"):
            del self.information_matrix_
        self.coef_ = coef_.ravel()
        self.se_ = se_
        self._k_factors_ = factor_data.shape[1]
        self.is_fitted = True

    def fit_mle(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray = None,
        no_intercepts: bool = False,
        no_factors: bool = False,
        no_spillovers: bool = False,
        # left_constrained: bool = False,
        # penalty_factor: float = 0,
        unconstrained: bool = False,
    ) -> None:
        """Fits the coefficients using maximum likelihood estimation.

        Args:
            return_data: (t_periods, n_series) array with return observations.
            factor_data: (t_periods, k_factors) array with factor observations.
        """

        # starting values
        assert self.n_series == return_data.shape[1], "data dimension does not match network"
        self.fit_wls(return_data=return_data, factor_data=factor_data)
        x0 = self.coef_
        x0[-1] /= 2
        # if not any([self.left_scaling, self.right_scaling]):
        #     x0 = self.coef_
        # else:
        #     x0 = self.coef_[:-1]
        #     if self.left_scaling:
        #         x0 = np.concatenate([x0, anp.full(self.n_series, self.coef_[-1])])

        #         # x0 = np.concatenate([x0, self.coef_[-1] * return_data.std().values * 10])
        #         # x0 = np.concatenate([x0, self.coef_[-1] / self.spillover_proxy.sum(axis=1) /2])
        #         # x0 = np.concatenate([x0, anp.ones((self.n_series))])
        #     if self.right_scaling:
        #         x0 = np.concatenate([x0, anp.ones((self.n_series))])

        # constraints
        bounds = [(None, None)] * (len(x0))
        bounds[-1] = (None, self.scaling_constraint * (0.9999))
        # if not any([self.left_scaling, self.right_scaling]):
        #     bounds[-1] = (None, self.scaling_constraint * (0.999))
        # if left_constrained:
        #     bounds[(1+self.k_factors_)*self.n_series:(2+self.k_factors_)*self.n_series] = [(None, 1/v) for v in self.spillover_proxy.sum(axis=1)]
        if no_intercepts:
            bounds[:self.n_series] = [(0, 0)] * self.n_series
        if no_factors:
            bounds[self.n_series:(1+self.k_factors_)*self.n_series] = [(0, 0)] * (self.n_series * self.k_factors_)
        if no_spillovers:
            bounds[(1+self.k_factors_)*self.n_series:] = [(0, 0)] * (len(x0) - (1+self.k_factors_)*self.n_series)
        if unconstrained:
            bounds[-1] = (None, None)

        # objective function
        def _neg_log_likelihood(params: anp.ndarray) -> float:
            # separate parameters
            T = return_data.shape[0]
            alpha = params[:self.n_series].reshape(self.n_series, 1)
            beta = params[self.n_series:(1+self.k_factors_)*self.n_series].reshape(self.n_series, self.k_factors_)
            c = params[(1+self.k_factors_)*self.n_series]
            D = c * self.spillover_proxy

            # if self.left_scaling:
            #     c1 = anp.diag(params[(1+self.k_factors_)*self.n_series:(2+self.k_factors_)*self.n_series])
            #     if self.right_scaling:
            #         c2 = anp.diag(params[-self.n_series:])
            #         D = c1 @ self.spillover_proxy @ c2
            #     else:
            #         D = c1 @ self.spillover_proxy
            # else:
            #     if self.right_scaling:
            #         c2 = anp.diag(params[-self.n_series:])
            #         D = self.spillover_proxy @ c2
            #     else:
            #         c = params[(1+self.k_factors_)*self.n_series]
            #         D = c * self.spillover_proxy

            # errors
            A = anp.eye(self.n_series) - D
            A_inv = anp.linalg.inv(A)
            E = anp.array(
                    A @ return_data.values.T
                    - alpha @ anp.ones((1, T))
                    - beta @ factor_data.values.T
                )
            sigma_eta = anp.diag(anp.diag((E @ E.T))) / T

            # log-likelihood
            ll = (
                - return_data.size/2 * anp.log(2*anp.pi)
                - T/2 * anp.linalg.slogdet(A_inv @ sigma_eta @ A_inv.T)[1]
                - 1/2 * anp.trace(anp.linalg.inv(sigma_eta) @ E @ E.T)
            )

            # FULL LUETKEPOHL (IDENTICAL SOLUTION)
            # E = anp.array(
            #         return_data.values.T
            #         - A_inv @ alpha @ anp.ones((1, T))
            #         - A_inv @ beta @ factor_data.values.T
            #     )
            # sigma_e = (E @ E.T) / T
            # sigma_eta = anp.diag(anp.diag((A @ sigma_e @ A.T)))
            # log-likelihood
            # ll = (
            #     - return_data.size/2 * anp.log(2*anp.pi)
            #     - T/2 * anp.linalg.slogdet(A_inv @ sigma_eta @ A_inv.T)[1]
            #     - 1/2 * anp.trace(E.T @ anp.linalg.inv(A_inv @ sigma_eta @ A_inv.T) @ E)
            # )

            # if penalty_factor != 0:
            #     assert not any([self.left_scaling, self.right_scaling]), "use penalty only for scalar scaling"
            #     log_penalty = - c/self.scaling_constraint * anp.log(1-c/self.scaling_constraint)
            #     ll -= penalty_factor * log_penalty

            return -ll

        # estimate
        jac = jacobian(_neg_log_likelihood)
        hess = hessian(_neg_log_likelihood)
        res = sp.optimize.minimize(fun=_neg_log_likelihood, x0=x0, options={"maxiter": 1000}, method="TNC", jac=jac, bounds=bounds)
        self.res_ = res
        if not res.success:
            warnings.warn(f"maximum likelihood estimation unsuccessful after {res.nit} iterations: {res.message}")

        # self.loglike_ = -res.fun
        # if penalty_factor != 0:
        #     constraint = 1/abs(anp.linalg.eigvals(self.spillover_proxy)).max()
        #     log_penalty = -res.x[-1]/constraint * anp.log(1-min(res.x[-1]/constraint, np.nextafter(1, 0)))
        #     self.loglike_ = -res.fun + log_penalty
        # else:
        #     self.loglike_ = -res.fun
        # self.aic_ = 2 * ((res.x!=0).sum() - self.loglike_)

        # store coefficient estimates
        if hasattr(self, "se_"):
            del self.se_
        self.coef_ = res.x
        self.information_matrix_ = np.linalg.inv(hess(res.x))
        self._k_factors_ = factor_data.shape[1]
        self.is_fitted = True

    def fit(
        self,
        method: str = "OLS",
        **kwargs,
    ) -> None:
        """Fit the FactorNetworkModel through one of the available fit methods.

        Available fitting methods are:
            OLS
            WLS
            MLE

        Args:
            method: Indicates the fitting method to use.
        """
        if method == "OLS":
            self.fit_ols(**kwargs)
        elif method == "WLS":
            self.fit_wls(**kwargs)
        elif method == "MLE":
            self.fit_mle(**kwargs)
        else:
            raise Exception("Fit method not implemented.")
    
    @property
    def wald_tests_(self) -> tuple:
        """Perfom wald tests for restrictions on individual parameters.
        
        Returns:
            w: The values of the test statistics.
            p_vals: The p-values of the test statistics.
        """
        w = self.coef_**2 / np.diag(self.information_matrix_)
        p_vals = sp.stats.chi2.cdf(w, df=1)
        return (w, p_vals)
    
    def predict(self, return_data: np.ndarray, factor_data: np.ndarray) -> pd.DataFrame:
        """Calculate fitted values for the fitted model and input data.

        Args:
            return_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            y_pred: (t_periods, n_series) dataframe with fitted values.
        """
        # setup
        t_periods = return_data.shape[0]
        n_series = return_data.shape[1]
        X = self._build_X(
            return_data=return_data, factor_data=factor_data,
        )

        # make predictions
        y_pred = (X @ self.coef_).reshape(t_periods, n_series, order="F")
        if type(return_data) == pd.DataFrame:
            y_pred = pd.DataFrame(
                y_pred, index=return_data.index, columns=return_data.columns
            )
        return y_pred

    def residuals(self, return_data: np.ndarray, factor_data: np.ndarray) -> pd.DataFrame:
        """Calculate prediction residuals for the fitted model and input data.

        Args:
            return_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            residuals: (t_periods, n_series) dataframe with residuals.
        """
        y_pred = self.predict(return_data=return_data, factor_data=factor_data)
        residuals = return_data - y_pred
        return residuals

    def r2(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weighting: str = "equal",
    ) -> float:
        """Calculate goodness of fit for the fitted model and input data.

        Args:
            return_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            weighting: Indicates how to weigh dependent variables. Available
                options are ["equal", "volatility", "granular"].

        Returns:
            r2: The goodness of fit statistic for the input data and the model.
        """
        levels = return_data.mean().values
        tss = np.nansum((return_data - levels) ** 2, axis=0)
        rss = np.nansum(self.residuals(return_data=return_data, factor_data=factor_data) ** 2, axis=0)
        if weighting == "equal":
            r2 = 1 - rss.sum() / tss.sum()
        elif weighting == "volatility":
            volatilities = return_data.std().values
            r2 = 1 - (rss / volatilities).sum() / (tss / volatilities).sum()
        elif weighting == "granular":
            r2 = 1 - rss / tss
        else:
            raise ValueError("weighting method '{}' not available".format(weighting))
        return r2

    def copy(self):
        """Returns a copy of the FactoNetworkModel object."""
        return copy.deepcopy(self)

    def factor_predict(
        self,
        factor_data: np.ndarray = None,
    ):
        """Calculate fitted values from the fitted factor loadings and input data.

        Args:
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            factor_predictions: (t_periods, n_series) dataframe with fitted values.
        """
        factor_predictions = factor_data @ self.factor_loadings_.T
        if self.has_intercepts:
            factor_predictions += self.intercepts_.T
        return factor_predictions

    def factor_residuals(
        self,
        return_data: np.ndarray = None,
        factor_data: np.ndarray = None,
    ):
        """Calculate prediction residuals from the fitted factor loadings and input data.

        Args:
            return_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            residuals: (t_periods, n_series) dataframe with factor residuals.
        """
        factor_predictions = self.factor_predict(factor_data=factor_data)
        if type(return_data) == pd.DataFrame:
            residuals = pd.DataFrame(
                return_data.values - factor_predictions.values,
                index=return_data.index,
                columns=return_data.columns,
            )
        return residuals

    # def systematic_variances(        WORKS BUT QUESTIONABLE IN THIS FRAMEWORK
    #     self,
    #     factor_data: np.ndarray = None,
    # ):
    #     """Calculate systematic variances from the fitted factor loadings and factor data.

    #     Args:
    #         factor_data: (t_periods, k_factors) array with factor observations.

    #     Returns:
    #         systematic_variances: The systematic variances of each series.
    #     """
    #     systematic_variances = self.factor_loadings_**2 @ factor_data.cov().values
    #     return systematic_variances

    def partial_r2s(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weighting: str = "equal",
    ) -> dict:
        """Calculate partial goodness of fit for each factor, all factors, and the VAR.

        Args:
            return_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            weighting: Indicates how to weigh dependent variables. Available
                options are ["equal", "volatility", "granular"].

        Returns:
            partial_r2s: The partial goodness of fit statistics.
        """

        # full model
        residuals = self.residuals(return_data=return_data, factor_data=factor_data)
        ss_full = np.nansum((residuals - residuals.mean()) ** 2, axis=0)

        # single factor restricted models
        ss_partial = []
        for i_factor in range(factor_data.shape[1]):
            restricted_model = self.copy()
            restricted_model.factor_loadings_[:, i_factor] = 0
            residuals = restricted_model.residuals(
                return_data=return_data, factor_data=factor_data
            )
            ss_partial += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # no factors model
        restricted_model = self.copy()
        restricted_model.factor_loadings_[:] = 0
        residuals = restricted_model.residuals(
            return_data=return_data, factor_data=factor_data
        )
        ss_partial += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # no spillovers model
        restricted_model = self.copy()
        restricted_model.coef_[-1] = 0
        residuals = restricted_model.residuals(
            return_data=return_data, factor_data=factor_data
        )
        ss_partial += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # calculate partial r2
        partial_r2s = []
        if weighting == "equal":
            for ss_restricted in ss_partial:
                partial_r2s += [1 - ss_full.sum() / ss_restricted.sum()]
        elif weighting == "volatility":
            for ss_restricted in ss_partial:
                volatilities = return_data.std().values
                partial_r2s += [
                    1 - (ss_full / volatilities).sum() / (ss_restricted / volatilities).sum()
                ]
        elif weighting == "granular":
            for ss_restricted in ss_partial:
                partial_r2s += [1 - ss_full / ss_restricted]
        else:
            raise ValueError("weighting method '{}' not available".format(weighting))

        # create output
        if type(factor_data) == pd.DataFrame:
            keys = factor_data.columns.tolist()
        else:
            keys = [f"factor_{i}" for i in range(factor_data.shape[1])]
        keys += ["factors", "spillovers"]
        partial_r2s = {k: v for (k, v) in zip(keys, partial_r2s)}
        return partial_r2s

    def component_r2s(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weighting: str = "equal",
    ) -> dict:
        """Calculate goodness of fit for each factor, all factors, and the VAR.

        Args:
            return_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            weighting: Indicates how to weigh dependent variables. Available
                options are ["equal", "volatility", "granular"].

        Returns:
            partial_r2s: The partial goodness of fit statistics.
        """

        # total variation
        levels = return_data.mean().values
        tss = np.nansum(
            (return_data - levels) ** 2,
            axis=0,
        )

        # single factor models
        ss_component = []
        for i_factor in range(factor_data.shape[1]):
            component_model = self.copy()
            component_model.factor_loadings_[:] = 0
            component_model.coef_[-1] = 0
            component_model.factor_loadings_[:, i_factor] = self.factor_loadings_[
                :, i_factor
            ]
            residuals = component_model.residuals(
                return_data=return_data, factor_data=factor_data
            )
            ss_component += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # pure factor model
        component_model = self.copy()
        component_model.coef_[-1] = 0
        residuals = component_model.residuals(
            return_data=return_data, factor_data=factor_data
        )
        ss_component += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # pure spillover model
        component_model = self.copy()
        component_model.factor_loadings_[:] = 0
        residuals = component_model.residuals(
            return_data=return_data, factor_data=factor_data
        )
        ss_component += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # calculate partial r2
        component_r2s = []
        if weighting == "equal":
            for ss_restricted in ss_component:
                component_r2s += [1 - ss_restricted.sum() / tss.sum()]
        elif weighting == "volatility":
            for ss_restricted in ss_component:
                volatilities = return_data.std().values
                component_r2s += [
                    1 - (ss_restricted / volatilities).sum() / (tss / volatilities).sum()
                ]
        elif weighting == "granular":
            for ss_restricted in ss_component:
                component_r2s += [1 - ss_restricted / tss]
        else:
            raise ValueError("weighting method '{}' not available".format(weighting))

        # create output
        if type(factor_data) == pd.DataFrame:
            keys = factor_data.columns.tolist()
        else:
            keys = [f"factor_{i}" for i in range(factor_data.shape[1])]
        keys += ["factors", "spillovers"]
        component_r2s = {k: v for (k, v) in zip(keys, component_r2s)}
        return component_r2s

    def residual_covariance(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
    ) -> pd.DataFrame:
        """"""
        residuals = self.residuals(return_data=return_data, factor_data=factor_data)
        sigma = residuals.cov()
        return np.diag(np.diag(sigma))
    
    def common_risk_premia(
        self,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        Sigma = factor_data.cov()
        arp = weights.T @ self.inv_structural_matrix_ @ self.factor_loadings_ @ np.diag(np.diag(Sigma))
        return arp
    
    def market_common_exposures(
        self,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        common_exp = weights.T @ self.adjusted_factor_loadings_
        return common_exp
    
    def expected_common_returns(
        self,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        common_ret = self.adjusted_factor_loadings_ @ self.common_risk_premia(factor_data=factor_data, weights=weights)
        return common_ret
    
    def granular_risk_premia(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        Sigma = self.residual_covariance(return_data=return_data, factor_data=factor_data)

        grp = weights.T @ self.inv_structural_matrix_ @ np.diag(np.diag(Sigma))
        return grp

    def market_granular_exposures(
        self,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        granular_exp = weights.T @ self.inv_structural_matrix_
        return granular_exp
    
    def expected_granular_returns(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        granular_ret = self.inv_structural_matrix_ @ self.granular_risk_premia(return_data=return_data, factor_data=factor_data, weights=weights)
        return granular_ret
    
    def expected_granular_returns_self(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        granular_ret = np.diag(np.diag(self.inv_structural_matrix_)) @ self.granular_risk_premia(return_data=return_data, factor_data=factor_data, weights=weights)
        return granular_ret
    
    def expected_granular_returns_others(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        granular_ret = (self.inv_structural_matrix_-np.diag(np.diag(self.inv_structural_matrix_))) @ self.granular_risk_premia(return_data=return_data, factor_data=factor_data, weights=weights)
        return granular_ret
    
    def expected_returns(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> np.ndarray:
        """"""
        common_ret = self.expected_common_returns(factor_data=factor_data, weights=weights)
        granular_ret = self.expected_granular_returns(return_data=return_data, factor_data=factor_data, weights=weights)
        return common_ret + granular_ret

    @property
    def log_likelihood_(self):
        """The log-likelihood of the fitted model."""
        return -self.res_.fun
    
    @property
    def num_params_(self):
        """The log-likelihood of the fitted model."""
        return (self.coef_!=0).sum() + self.n_series
    
    @property
    def aic_(self):
        """The log-likelihood of the fitted model."""
        return 2 * (self.num_params_ - self.log_likelihood_)
    

    def asset_variance_decomposition(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
    ) -> list:
        """"""
        Sigma_delta = factor_data.cov()
        Sigma_eta = self.residual_covariance(return_data=return_data, factor_data=factor_data)
        
        Gamma_plus = np.diag(np.diag(self.inv_structural_matrix_))
        Gamma_minus = self.inv_structural_matrix_ - Gamma_plus

        common = self.adjusted_factor_loadings_ @ Sigma_delta @ self.adjusted_factor_loadings_.T
        spillover = Gamma_minus @ Sigma_eta @ Gamma_minus.T + Gamma_plus @ Sigma_eta @ Gamma_minus.T + Gamma_minus @ Sigma_eta @ Gamma_plus.T
        idiosyncratic = Gamma_plus @ Sigma_eta @ Gamma_plus.T

        return [np.array(common), np.array(spillover), np.array(idiosyncratic)]
    
    def market_variance_decomposition(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
        weights: np.ndarray = np.ones((100, 1)),
    ) -> list:
        """"""
        common, spillover, idiosyncratic = self.asset_variance_decomposition(return_data=return_data, factor_data=factor_data)
        common = weights.T @ common @ weights
        spillover = weights.T @ spillover @ weights
        idiosyncratic = weights.T @ idiosyncratic @weights
        return [common, spillover, idiosyncratic]
    
    def lr_test(
        self,
        return_data: np.ndarray,
        factor_data: np.ndarray,
    ) -> tuple:
        """Likelihood ratio test for overidentifying restrictions."""
        N = self.n_series

        Sigma_reduced = self.factor_residuals(return_data=return_data, factor_data=factor_data).cov().values

        Sigma_eta = self.residual_covariance(return_data=return_data, factor_data=factor_data)
        Sigma_restricted = self.inv_structural_matrix_ @ Sigma_eta @ self.inv_structural_matrix_.T

        lr = len(return_data) * (np.linalg.slogdet(Sigma_restricted)[1] - np.linalg.slogdet(Sigma_reduced)[1])

        p_val = 1 - sp.stats.chi2.cdf(lr, df=N*(N+1)/2-self.num_params_)
        return (lr, p_val)
        