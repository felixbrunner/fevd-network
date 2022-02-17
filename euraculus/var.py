import numpy as np
import scipy as sp
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.covariance import LedoitWolf, GraphicalLassoCV, OAS

from euraculus.net import ElasticNet, AdaptiveElasticNet

import copy


class VAR:
    """Vector auto-regression."""

    def __init__(self, add_intercepts=True, p_lags=1):
        self.add_intercepts = add_intercepts
        self.p_lags = p_lags

    def fit(self, X, y=None, exog=None, method="OLS", **kwargs):
        """Wrapper for available fit methods:
        - OLS
        - ElasticNet
        - AdaptiveElasticNet
        - ElasticNetCV
        - AdaptiveElasticNetCV
        """
        if method == "OLS":
            self.fit_ols(X, y=None, exog=None, **kwargs)
        elif method == "ElasticNet":
            self.fit_elastic_net(X, y=None, exog=None, **kwargs)
        elif method == "AdaptiveElasticNet":
            self.fit_adaptive_elastic_net(X, y=None, exog=None, **kwargs)
        elif method == "ElasticNetCV":
            self.fit_elastic_net_cv(X, y=None, exog=None, **kwargs)
        elif method == "AdaptiveElasticNetCV":
            self.fit_adaptive_elastic_net_cv(X, y=None, exog=None, **kwargs)
        else:
            raise Exception("Fit method not implemented.")

    def _build_y(self, var_data):
        """Returns a numpy array containing the dependent
        variable reshaped for estimation.
        """
        y = var_data.values[self.p_lags :].reshape(-1, 1, order="F")
        return y

    def _build_X_block(self, var_data, add_intercepts=True, exog_data=None):
        """Returns a numpy array consisting of:
        - a vector of ones
        - the series data for VAR estimation
        """
        # setup
        t_periods = var_data.shape[0] - self.p_lags
        elements = []

        # constant
        if add_intercepts:
            elements += [np.ones([t_periods, 1])]

        # exogenous regressors
        if exog_data is not None:
            assert (
                var_data.shape[0] == exog_data.shape[0]
            ), "exog shape does not match data"
            elements += [exog_data[self.p_lags - 1 : -1]]

        # first lag
        elements += [var_data[self.p_lags - 1 : -1]]

        # higher lags
        if self.p_lags > 1:
            for l in range(self.p_lags - 1):
                elements += [var_data[self.p_lags - 2 - l : -2 - l]]

        # build block
        X_block = np.concatenate(elements, axis=1)
        return X_block

    def _build_X(self, var_data, add_intercepts=True, exog_data=None):
        """Returns sparse diagonal block matrix with independent variables."""
        # size
        n_series = var_data.shape[1]

        # build
        X_block = self._build_X_block(
            var_data, add_intercepts=add_intercepts, exog_data=exog_data
        )
        X = sp.sparse.kron(sp.sparse.eye(n_series), X_block, format="csc")
        return X

    def _build_X_y(self, var_data, add_intercepts=True, exog_data=None):
        """Returns data matrices for estimation"""
        X = self._build_X(var_data, add_intercepts=add_intercepts, exog_data=exog_data)
        y = self._build_y(var_data)
        return (X, y)

    def fit_ols(self, X, y=None, exog=None, return_model=False):
        """Fits the VAR coefficients using OLS."""
        # build inputs
        var_data = X.copy()
        X, y = self._build_X_y(
            var_data, add_intercepts=self.add_intercepts, exog_data=exog
        )

        # estimate
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        # store coefficient estimates
        self.coef_ = model.coef_.ravel()
        self._split_coef(var_data, exog_data=exog)

        # returns
        if return_model:
            return model

    def _split_coef(self, var_data, exog_data=None):
        """Splits the coefficient vector into its components:
        - intercepts
        - exogenous regressor coefficients
        - VAR coefficient matrices
        """
        # dimensions
        a_const = int(self.add_intercepts)
        if exog_data is not None:
            k_exog = exog_data.shape[1]
        else:
            k_exog = 0
        n_series = var_data.shape[1]
        p_lags = self.p_lags
        m_features = a_const + k_exog + n_series * p_lags

        # split & store
        self.intercepts_ = self._extract_intercepts(
            m_features=m_features, n_series=n_series, a_const=a_const
        )
        self.exog_loadings_ = self._extract_exog_loadings(
            m_features=m_features, k_exog=k_exog, n_series=n_series, a_const=a_const
        )
        self.var_matrices_ = self._extract_var_matrices(
            m_features=m_features, n_series=n_series, k_exog=k_exog, a_const=a_const
        )

    def _extract_intercepts(self, m_features, n_series, a_const):
        """Returns a numpy array of intercepts."""
        indices = list(
            np.concatenate(
                [np.arange(0, 0 + a_const) + m_features * i for i in range(n_series)]
            )
        )
        intercepts = self.coef_[indices].reshape(-1, 1)
        return intercepts

    def _extract_exog_loadings(self, a_const, m_features, k_exog, n_series):
        """Returns a numpy array of exogenous loadings."""
        indices = list(
            np.concatenate(
                [
                    np.arange(a_const, a_const + k_exog) + m_features * i
                    for i in range(n_series)
                ]
            )
        )
        beta_matrix = self.coef_[indices].reshape(n_series, k_exog)
        return beta_matrix

    def _extract_var_matrices(self, m_features, n_series, k_exog, a_const):
        """Returns a list of numpy arrays with VAR coefficients."""
        var_matrices = []
        for l in range(self.p_lags):
            indices = list(
                np.concatenate(
                    [
                        np.arange(a_const + k_exog, a_const + k_exog + n_series)
                        + m_features * i
                        + n_series * l
                        for i in range(n_series)
                    ]
                )
            )
            var_matrix = self.coef_[indices].reshape(n_series, n_series)
            var_matrices += [var_matrix]
        return var_matrices

    @property
    def var_1_matrix_(self):
        """Returns a numpy array of first lag VAR coefficients."""
        var_matrix = self.var_matrices_[0]
        return var_matrix

    def predict(self, var_data, exog_data=None):
        """Returns fitted values for a trained model
        and input data.
        """
        # setup
        t_periods = var_data.shape[0] - self.p_lags
        n_series = var_data.shape[1]
        X = self._build_X(
            var_data, add_intercepts=self.add_intercepts, exog_data=exog_data
        )

        # make predictions
        y_pred = (X @ self.coef_).reshape(t_periods, n_series, order="F")
        y_pred = pd.DataFrame(
            y_pred, index=var_data.index[self.p_lags :], columns=var_data.columns
        )
        return y_pred

    def residuals(self, var_data, exog_data=None):
        """Returns the prediction residuals for
        a trained model and input data.
        """
        y_pred = self.predict(var_data, exog_data=exog_data)
        residuals = var_data[self.p_lags :] - y_pred
        return residuals

    def exog_predict(self, exog_data):
        """NOT IMPLEMENTED, PLACEHOLDER"""
        pass

    def exog_residuals(self, exog_data):
        """NOT IMPLEMENTED, PLACEHOLDER"""
        pass

    def r2(self, var_data, exog_data=None):
        """Goodness of fit statistic for a trained model
        and input data.
        """
        levels = var_data[self.p_lags :].mean().values
        tss = np.nansum((var_data[self.p_lags :] - levels) ** 2)
        rss = np.nansum(self.residuals(var_data, exog_data=exog_data) ** 2)
        r2 = 1 - rss / tss
        return r2

    @property
    def df_used_(self):
        """Returns the number of degrees of freedom
        used in the estimation.
        """
        df_used = (
            np.count_nonzero(self.intercepts_)
            + np.count_nonzero(self.exog_loadings_)
            + np.count_nonzero(self.var_matrices_)
        )
        return df_used

    @property
    def coef_density_(self):
        """Returns the estimate density of a trained model."""
        density = self.df_used_ / len(self.coef_)
        return density

    @property
    def coef_sparsity_(self):
        """Returns the estimate sparsity of a trained model."""
        sparsity = 1 - self.coef_density_
        return sparsity

    @property
    def var_density_(self):
        """Returns the estimate density of a trained model."""
        density = np.count_nonzero(self.var_matrices_) / np.size(self.var_matrices_)
        return density

    @property
    def var_sparsity_(self):
        """Returns the estimate sparsity of a trained model."""
        sparsity = 1 - self.var_density_
        return sparsity

    def _scale_data(self, data, demean=True):
        """Scales input data to be centered at zero, with
        standard deviation of one. Centering is optional.
        """
        if data is None:
            return None
        # fit
        levels = data.mean().values
        scales = data.std().values

        # transform
        scaled_data = (data - levels) / scales
        if not demean:
            scaled_data += levels

        return scaled_data

    def _scale_coefs(self, model, var_data, exog_data=None):
        """Recovers the original scale of the data after estimating the
        coefficients on a standardised version of the data.
        """
        # dimensions
        if exog_data is not None:
            k_exog = exog_data.shape[1]
        else:
            k_exog = 0
        n_series = var_data.shape[1]
        m_features = k_exog + n_series * self.p_lags

        # construct inputs
        y_levels = var_data.mean().values.reshape(n_series, 1)
        y_scales = var_data.std().values.reshape(n_series, 1)
        x_block = self._build_X_block(
            var_data, add_intercepts=False, exog_data=exog_data
        )
        x_levels = x_block.mean(axis=0).reshape(1, m_features)
        x_scales = x_block.std(axis=0).reshape(1, m_features)

        # rescale coefficients
        coef_block = model.coef_.reshape(n_series, m_features)
        scaled_coef_block = coef_block * y_scales / x_scales

        # add intercepts
        if self.add_intercepts:
            intercepts = y_levels - scaled_coef_block @ x_levels.T
            scaled_coef_block = np.concatenate([intercepts, scaled_coef_block], axis=1)

        # generate output
        scaled_coef = scaled_coef_block.ravel()
        return scaled_coef

    def fit_elastic_net(
        self,
        X,
        y=None,
        exog=None,
        alpha=0.1,
        lambdau=0.1,
        penalty_weights=None,
        return_model=False,
        **kwargs
    ):
        """Fits the VAR coefficients using the glmnet routine."""
        # build inputs
        var_data = X.copy()
        scaled_var_data = self._scale_data(var_data, demean=self.add_intercepts)
        scaled_exog = self._scale_data(exog, demean=self.add_intercepts)
        X, y = self._build_X_y(
            scaled_var_data, add_intercepts=False, exog_data=scaled_exog
        )

        # estimate
        model = ElasticNet(
            alpha=alpha, lambdau=lambdau, intercept=False, standardize=False, **kwargs
        )
        model.fit(X, y, penalty_weights=penalty_weights)

        # store estimates
        self.coef_ = self._scale_coefs(model=model, var_data=var_data, exog_data=exog)
        self._split_coef(var_data, exog_data=exog)

        # returns
        if return_model:
            return model

    def fit_adaptive_elastic_net(
        self,
        X,
        y=None,
        exog=None,
        alpha=0.1,
        lambdau=0.1,
        penalty_weights=None,
        return_model=False,
        **kwargs
    ):
        """Fits the VAR coefficients using the adaptive elastic net."""
        # build inputs
        var_data = X.copy()
        scaled_var_data = self._scale_data(var_data, demean=self.add_intercepts)
        scaled_exog = self._scale_data(exog, demean=self.add_intercepts)
        X, y = self._build_X_y(
            scaled_var_data, add_intercepts=False, exog_data=scaled_exog
        )

        # estimate
        model = AdaptiveElasticNet(
            alpha=alpha, lambdau=lambdau, intercept=False, standardize=False, **kwargs
        )
        model.fit(X, y, penalty_weights=penalty_weights)

        # store estimates
        self.coef_ = self._scale_coefs(model=model, var_data=var_data, exog_data=exog)
        self._split_coef(var_data, exog_data=exog)

        # returns
        if return_model:
            return model

    def _make_cv_splitter(self, var_data, folds=12):
        """Returns a PredefinedSplit object for cross validation."""
        # shapes
        t_periods = var_data.shape[0] - self.p_lags
        n_series = var_data.shape[1]
        length = t_periods // folds
        resid = t_periods % folds

        # build single series split
        single_series_split = []
        for i in range(folds - 1, -1, -1):
            single_series_split += length * [i]
            if i < resid:
                single_series_split += [i]

        # make splitter object
        split = n_series * single_series_split
        splitter = PredefinedSplit(split)
        return splitter

    def fit_elastic_net_cv(
        self, X, y=None, exog=None, grid=None, folds=12, return_cv=False, **kwargs
    ):
        """Fits the VAR coefficients using the glmnet routine.
        Sets the model coefficients of the best estimator.
        return_cv can be set to True to return CV object.
        """
        # build inputs
        var_data = X.copy()
        scaled_var_data = self._scale_data(var_data, demean=self.add_intercepts)
        scaled_exog = self._scale_data(exog, demean=self.add_intercepts)
        X, y = self._build_X_y(
            scaled_var_data, add_intercepts=False, exog_data=scaled_exog
        )

        # set up CV
        split = self._make_cv_splitter(var_data, folds=folds)
        elnet = ElasticNet(intercept=False, standardize=False)

        # estimate
        cv = GridSearchCV(
            elnet,
            grid,
            cv=split,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            **kwargs
        )
        cv.fit(X, y, penalty_weights=None)

        # store estimates
        self.coef_ = self._scale_coefs(
            model=cv.best_estimator_, var_data=var_data, exog_data=exog
        )
        self._split_coef(var_data, exog_data=exog)

        # returns
        if return_cv:
            return cv

    def fit_adaptive_elastic_net_cv(
        self, X, y=None, exog=None, grid=None, folds=12, return_cv=False, **kwargs
    ):
        """Fits the VAR coefficients using the glmnet routine.
        Sets the model coefficients of the best estimator.
        return_cv can be set to True to return CV object.
        """
        # build inputs
        var_data = X.copy()
        scaled_var_data = self._scale_data(var_data, demean=self.add_intercepts)
        scaled_exog = self._scale_data(exog, demean=self.add_intercepts)
        X, y = self._build_X_y(
            scaled_var_data, add_intercepts=False, exog_data=scaled_exog
        )

        # set up CV
        split = self._make_cv_splitter(var_data, folds=folds)
        elnet = AdaptiveElasticNet(intercept=False, standardize=False)
        elnet.fit(
            X, y, ini_split=split
        )  # required to update the penalty weights only once

        # estimate
        cv = GridSearchCV(
            elnet,
            grid,
            cv=split,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            **kwargs
        )
        cv.fit(X, y, split=split)

        # store estimates
        self.coef_ = self._scale_coefs(
            model=cv.best_estimator_, var_data=var_data, exog_data=exog
        )
        self._split_coef(var_data, exog_data=exog)

        # returns
        if return_cv:
            return cv

    def _penalty_weights(self):
        """NOT IMPLEMENTED, PLACEHOLDER
        To exclude diagonal entries in the VAR matrices from penalisation.
        """
        pass

    def copy(self):
        """Returns a copy of self"""
        return copy.deepcopy(self)


###################################################################################################


class OldVAR:
    """Vector autoregression."""

    def __init__(self, var_data, p_lags):
        self.var_data = var_data
        self.p_lags = p_lags

    @property
    def t_periods(self):
        """Returns the number of periods used for estimation."""
        t_periods = self.var_data.shape[0] - self.p_lags
        return t_periods

    @property
    def n_series(self):
        """Returns the number of series in the VAR."""
        n_series = self.var_data.shape[1]
        return n_series

    def _build_y(self):
        """Returns a numpy array containing the dependent
        variable reshaped for estimation.
        """
        y = self.var_data.values[self.p_lags :].reshape(-1, 1, order="F")
        return y

    def _build_X_block(self):
        """Returns a numpy array consisting of:
        - a vector of ones
        - the series data for VAR estimation
        """
        # constant
        ones = np.ones([self.t_periods, 1])

        # first lag
        var = self.var_data[self.p_lags - 1 : -1]

        # higher lags
        if self.p_lags > 1:
            for l in range(self.p_lags - 1):
                var = np.concatenate(
                    [var, self.var_data[self.p_lags - 2 - l : -2 - l]], axis=1
                )

        # block
        X_block = np.concatenate([ones, var], axis=1)
        return X_block

    def _build_X(self):
        """Returns sparsediagonal block matrix with independent variables."""
        # build
        X_block = self._build_X_block()
        X = sp.sparse.kron(sp.sparse.eye(self.n_series), X_block, format="csc")
        return X

    def _build_X_y(self):
        """Returns data matrices for estimation"""
        X = self._build_X()
        y = self._build_y()
        return (X, y)

    def fit_OLS(self, return_model=False):
        """Fits the FAVAR coefficients using OLS."""
        # build inputs
        X, y = self._build_X_y()

        # estimate
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        if return_model:
            return model
        else:
            self.coef_ = model.coef_.ravel()

    def fit_elastic_net(
        self, alpha, lambdau, penalty_weights=None, return_model=False, **kwargs
    ):
        """Fits the FAVAR coefficients using the glmnet routine."""
        # build inputs
        X, y = self._build_X_y()

        # estimate
        model = ElasticNet(
            alpha=alpha, lambdau=lambdau, intercept=False, standardize=False, **kwargs
        )
        model.fit(X, y, penalty_weights=penalty_weights)

        if return_model:
            return model
        else:
            self.coef_ = model.coef_.ravel()
            self.r2 = model.R2_
            self.df_used = model.df_used_

    @property
    def m_features(self):
        """Returns the number of independent variables."""
        m_features = 1 + self.n_series * self.p_lags
        return m_features

    def _penalty_weights(self, method="OLS", penalise_diagonal=True, **kwargs):
        """Returns estimated penalty weights for VAR coefficients.
        Note that intercepts and factor loadings are not penalised.
        """
        # estimate
        if method == "OLS":
            coefficients = self.fit_OLS(return_model=True).coef_
        elif method == "ElasticNet":
            coefficients = (
                self.fit_elastic_net(return_model=True, **kwargs).coef_ + 1e-8
            )
        elif method == "ridge":
            coefficients = self.fit_elastic_net(return_model=True, **kwargs).coef_
        elif method == "equal":
            coefficients = np.ones(self.m_features * self.n_series)
        else:
            raise NotImplementedError("method not implemented.")

        # make list of non-penalised parameters
        zero_penalty_indices = [
            self.m_features * i for i in range(self.n_series)
        ]  # intercepts
        if not penalise_diagonal:
            zero_penalty_indices += [
                1 + (self.m_features + 1) * i for i in range(self.n_series)
            ]  # VAR1 diagonal

        # create output
        penalty_weights = abs(coefficients.ravel()) ** -1
        penalty_weights[zero_penalty_indices] = 0
        return penalty_weights

    @property
    def intercepts_(self):
        """Returns a numpy array of intercepts."""
        indices = [self.m_features * i for i in range(self.n_series)]
        intercepts = self.coef_[indices].reshape(-1, 1)
        return intercepts

    @property
    def var_1_matrix_(self):
        """Returns a numpy array of first lag VAR coefficients."""
        indices = list(
            np.concatenate(
                [
                    np.arange(1, 1 + self.n_series) + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )
        var_matrix = self.coef_[indices].reshape(self.n_series, self.n_series)
        return var_matrix

    @property
    def var_matrices_(self):
        """Returns a list of numpy arrays with VAR coefficients."""
        var_matrices = []
        for l in range(self.p_lags):
            indices = list(
                np.concatenate(
                    [
                        np.arange(1, 1 + self.n_series)
                        + self.m_features * i
                        + self.n_series * l
                        for i in range(self.n_series)
                    ]
                )
            )
            var_matrix = self.coef_[indices].reshape(self.n_series, self.n_series)
            var_matrices += [var_matrix]
        return var_matrices

    @property
    def residuals_(self):
        """Returns the residual matrix to a vectorised VAR model."""
        X, y = self._build_X_y()
        residuals = y - X.dot(self.coef_.reshape(-1, 1))
        residual_matrix = residuals.reshape(-1, self.n_series, order="F")
        return residual_matrix

    def make_cv_splitter(self, folds=12):
        """Returns a PredefinedSplit object for cross validation."""

        # shapes
        length = self.t_periods // folds
        resid = self.t_periods % folds

        # build single series split
        single_series_split = []
        for i in range(folds - 1, -1, -1):
            single_series_split += length * [i]
            if i < resid:
                single_series_split += [i]

        # make splitter object
        split = self.n_series * single_series_split
        splitter = PredefinedSplit(split)
        return splitter

    def fit_elastic_net_cv(
        self,
        grid,
        folds=12,
        return_cv=False,
        weighting_method="OLS",
        penalise_diagonal=True,
        **kwargs
    ):
        """Fits the FAVAR coefficients using the glmnet routine.
        Uses OLS inferred penalty weights.
        Sets the model coefficients as the best estimator.
        return_cv can be set to True to return CV object.
        """
        # build inputs
        X, y = self._build_X_y()
        self.fit_OLS()
        ols_estimate = self.var_1_matrix_
        if weighting_method == "equal":
            penalty_weights = self._penalty_weights(
                method="equal", penalise_diagonal=penalise_diagonal
            )
        elif weighting_method == "OLS":
            penalty_weights = self._penalty_weights(
                method="OLS", penalise_diagonal=penalise_diagonal
            )
        elif weighting_method == "ElasticNet":
            #             lambdau = ElasticNet(intercept=False, standardize=False).find_lambda(X, y, lambda_grid=grid['lambdau'], alpha=0.5)
            lambdau = ElasticNet(intercept=False, standardize=False).find_lambda(
                X,
                y,
                lambda_grid=np.unique(
                    np.concatenate(
                        [np.geomspace(1e-5, 1e5, 25), np.linspace(1e-5, 1e5, 25)]
                    )
                ),
                alpha=0.01,
            )
            penalty_weights = self._penalty_weights(
                method="ElasticNet",
                penalise_diagonal=penalise_diagonal,
                alpha=0.01,
                lambdau=lambdau,
            )
        elif weighting_method == "ridge":
            #             lambdau = ElasticNet(intercept=False, standardize=False).find_lambda(X, y, lambda_grid=grid['lambdau'], alpha=0)
            lambdau = ElasticNet(intercept=False, standardize=False).find_lambda(
                X,
                y,
                lambda_grid=np.unique(
                    np.concatenate(
                        [np.geomspace(1e-5, 1e5, 25), np.linspace(1e-5, 1e5, 25)]
                    )
                ),
                alpha=0,
            )
            penalty_weights = self._penalty_weights(
                method="ridge",
                penalise_diagonal=penalise_diagonal,
                alpha=0,
                lambdau=lambdau,
            )
        else:
            raise NotImplementedError("weighting method not implemented")
        split = self.make_cv_splitter(folds=folds)
        elnet = ElasticNet(intercept=False, standardize=False)

        # estimate
        cv = GridSearchCV(elnet, grid, cv=split, n_jobs=-1, verbose=1, **kwargs)
        cv.fit(X, y, penalty_weights=penalty_weights)

        self.coef_ = cv.best_estimator_.coef_.ravel()
        self.r2 = cv.best_estimator_.R2_
        self.df_used = cv.best_estimator_.df_used_
        self.mean_shrinkage_ = (
            1
            - (
                abs(self.var_1_matrix_[self.var_1_matrix_ != 0])
                / abs(ols_estimate[self.var_1_matrix_ != 0])
            ).mean()
        )
        if return_cv:
            return cv

    def get_lambda_path(self, nlambda=10, **kwargs):
        """Returns the lambda path chosen by the glmnet routine."""
        import glmnet_python
        from glmnet import glmnet

        X, y = self._build_X_y()
        penalty_weights = self._penalty_weights(method="OLS")
        fit = glmnet(
            x=X, y=y, nlambda=nlambda, penalty_factor=penalty_weights, **kwargs
        )
        return fit["lambdau"]


#######################################################################################################


class VARX(VAR):
    """Vector autoregression with exogenous regressors.
    Note that exogenous regressors are currently not lagged.
    """

    def __init__(self, var_data, exog_data, p_lags=1):
        self.var_data = var_data
        self.exog_data = exog_data
        self.p_lags = p_lags
        self._check_dims()

    @property
    def k_exog(self):
        """Returns the number of exogenous regressors used in the VARX."""
        k_exog = self.exog_data.shape[1]
        return k_exog

    def _build_X_block(self):
        """Returns a numpy array consisting of:
        - a vector of ones
        - the exogenous data
        - the series data for VAR estimation
        """
        # block components
        ones = np.ones([self.t_periods, 1])
        exog = self.exog_data[self.p_lags :]

        # first lag
        var = self.var_data[self.p_lags - 1 : -1]
        # higher lags
        if self.p_lags > 1:
            for l in range(self.p_lags - 1):
                var = np.concatenate(
                    [var, self.var_data[self.p_lags - 2 - l : -2 - l]], axis=1
                )

        # block
        X_block = np.concatenate([ones, exog, var], axis=1)
        return X_block

    def _build_X_y(self):
        """Returns data matrices for estimation"""
        self._check_dims()
        X = self._build_X()
        y = self._build_y()
        return (X, y)

    def _check_dims(self):
        """Checks if data shapes match."""
        assert (
            self.var_data.shape[0] == self.exog_data.shape[0]
        ), "data needs to have same time dimension"

    @property
    def m_features(self):
        """Returns the number of independent variables."""
        m_features = 1 + self.k_exog + self.n_series * self.p_lags
        return m_features

    def _penalty_weights(self, method="OLS", **kwargs):
        """Returns estimated penalty weights for VAR coefficients.
        Note that intercepts and factor loadings are not penalised.
        """
        # estimate
        if method == "OLS":
            coefficients = self.fit_OLS(return_model=True).coef_
        elif method == "ElasticNet":
            coefficients = self.fit_elastic_net(return_model=True, **kwargs).coef_
        else:
            raise NotImplementedError("method not implemented.")

        # make list of non-penalised parameters
        zero_penalty_indices = list(
            np.concatenate(
                [
                    np.arange(0, 1 + self.k_exog) + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )

        # create output
        penalty_weights = abs(coefficients.ravel()) ** -1
        penalty_weights[zero_penalty_indices] = 0
        return penalty_weights

    @property
    def exog_loadings_(self):
        """Returns a numpy array of exogenous loadingss."""
        indices = list(
            np.concatenate(
                [
                    np.arange(1, 1 + self.k_exog) + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )
        beta_matrix = self.coef_[indices].reshape(self.n_series, self.k_exog)
        return beta_matrix

    @property
    def var_1_matrix_(self):
        """Returns a numpy array of first lag VAR coefficients."""
        indices = list(
            np.concatenate(
                [
                    np.arange(1 + self.k_exog, 1 + self.k_exog + self.n_series)
                    + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )
        var_matrix = self.coef_[indices].reshape(self.n_series, self.n_series)
        return var_matrix

    @property
    def var_matrices_(self):
        """Returns a list of numpy arrays with VAR coefficients."""
        var_matrices = []
        for l in range(self.p_lags):
            indices = list(
                np.concatenate(
                    [
                        np.arange(1 + self.k_exog, 1 + self.k_exog + self.n_series)
                        + self.m_features * i
                        + self.n_series * l
                        for i in range(self.n_series)
                    ]
                )
            )
            var_matrix = self.coef_[indices].reshape(self.n_series, self.n_series)
            var_matrices += [var_matrix]
        return var_matrices

    @property
    def exog_residuals_(self):
        """Returns the residual matrix to a vectorised VAR model."""
        X, y = self._build_X_y()

        # factor model coefficients
        var_indices = list(
            np.concatenate(
                [
                    np.arange(
                        1 + self.k_exog, 1 + self.k_exog + self.n_series * self.p_lags
                    )
                    + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )
        betas = self.coef_.copy()
        betas[var_indices] = 0

        residuals = y - X.dot(betas.reshape(-1, 1))
        residual_matrix = residuals.reshape(-1, self.n_series, order="F")
        return residual_matrix


####################################################################################################################


class FAVAR:
    """Factor augmented vector autoregression."""

    def __init__(self, var_data, factor_data, p_lags=1):
        self.var_data = var_data
        self.factor_data = factor_data
        self.p_lags = p_lags
        self._check_dims()

    @property
    def t_periods(self):
        """Returns the number of periods used for estimation."""
        t_periods = self.var_data.shape[0] - self.p_lags
        return t_periods

    @property
    def n_series(self):
        """Returns the number of series in the VAR."""
        n_series = self.var_data.shape[1]
        return n_series

    @property
    def k_factors(self):
        """Returns the number of factors used in the FAVAR."""
        k_factors = self.factor_data.shape[1]
        return k_factors

    def _build_y(self):
        """Returns a numpy array containing the dependent
        variable reshaped for estimation.
        """
        y = self.var_data.values[self.p_lags :].reshape(-1, 1, order="F")
        return y

    def _build_X_block(self):
        """Returns a numpy array consisting of:
        - a vector of ones
        - the factors data
        - the series data for VAR estimation
        """
        # block components
        ones = np.ones([self.t_periods, 1])
        factors = self.factor_data[self.p_lags :]

        # first lag
        var = self.var_data[self.p_lags - 1 : -1]
        # higher lags
        if self.p_lags > 1:
            for l in range(self.p_lags - 1):
                var = np.concatenate(
                    [var, self.var_data[self.p_lags - 2 - l : -2 - l]], axis=1
                )

        # block
        X_block = np.concatenate([ones, factors, var], axis=1)
        return X_block

    def _build_X(self):
        """Returns sparsediagonal block matrix with independent variables."""
        # build
        X_block = self._build_X_block()
        X = sp.sparse.kron(sp.sparse.eye(self.n_series), X_block, format="csc")
        return X

    def _build_X_y(self):
        """Returns data matrices for estimation"""
        self._check_dims()
        X = self._build_X()
        y = self._build_y()
        return (X, y)

    def _check_dims(self):
        """Checks if data shapes match."""
        assert (
            self.var_data.shape[0] == self.factor_data.shape[0]
        ), "data needs to have same time dimension"

    def fit_OLS(self, return_model=False):
        """Fits the FAVAR coefficients using OLS."""
        # build inputs
        X, y = self._build_X_y()

        # estimate
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        if return_model:
            return model
        else:
            self.coef_ = model.coef_.ravel()

    def fit_elastic_net(
        self, alpha, lambdau, penalty_weights=None, return_model=False, **kwargs
    ):
        """Fits the FAVAR coefficients using the glmnet routine."""
        # build inputs
        X, y = self._build_X_y()

        # estimate
        model = ElasticNet(
            alpha=alpha, lambdau=lambdau, intercept=False, standardize=False, **kwargs
        )
        model.fit(X, y, penalty_weights=penalty_weights)

        if return_model:
            return model
        else:
            self.coef_ = model.coef_.ravel()

    @property
    def m_features(self):
        """Returns the number of independent variables."""
        m_features = 1 + self.k_factors + self.n_series * self.p_lags
        return m_features

    def _penalty_weights(self, method="OLS", **kwargs):
        """Returns estimated penalty weights for VAR coefficients.
        Note that intercepts and factor loadings are not penalised.
        """
        # estimate
        if method == "OLS":
            coefficients = self.fit_OLS(return_model=True).coef_
        elif method == "ElasticNet":
            coefficients = self.fit_elastic_net(return_model=True, **kwargs).coef_
        else:
            raise NotImplementedError("method not implemented.")

        # make list of non-penalised parameters
        zero_penalty_indices = list(
            np.concatenate(
                [
                    np.arange(0, 1 + self.k_factors) + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )

        # create output
        penalty_weights = abs(coefficients.ravel()) ** -1
        penalty_weights[zero_penalty_indices] = 0
        return penalty_weights

    @property
    def intercepts_(self):
        """Returns a numpy array of intercepts."""
        indices = [self.m_features * i for i in range(self.n_series)]
        intercepts = self.coef_[indices].reshape(-1, 1)
        return intercepts

    @property
    def factor_loadings_(self):
        """Returns a numpy array of factor loadingss."""
        indices = list(
            np.concatenate(
                [
                    np.arange(1, 1 + self.k_factors) + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )
        beta_matrix = self.coef_[indices].reshape(self.n_series, self.k_factors)
        return beta_matrix

    @property
    def var_1_matrix_(self):
        """Returns a numpy array of first lag VAR coefficients."""
        indices = list(
            np.concatenate(
                [
                    np.arange(1 + self.k_factors, 1 + self.k_factors + self.n_series)
                    + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )
        var_matrix = self.coef_[indices].reshape(self.n_series, self.n_series)
        return var_matrix

    @property
    def var_matrices_(self):
        """Returns a list of numpy arrays with VAR coefficients."""
        var_matrices = []
        for l in range(self.p_lags):
            indices = list(
                np.concatenate(
                    [
                        np.arange(
                            1 + self.k_factors, 1 + self.k_factors + self.n_series
                        )
                        + self.m_features * i
                        + self.n_series * l
                        for i in range(self.n_series)
                    ]
                )
            )
            var_matrix = self.coef_[indices].reshape(self.n_series, self.n_series)
            var_matrices += [var_matrix]
        return var_matrices

    @property
    def residuals_(self):
        """Returns the residual matrix to a vectorised VAR model."""
        X, y = self._build_X_y()
        residuals = y - X.dot(self.coef_.reshape(-1, 1))
        residual_matrix = residuals.reshape(-1, self.n_series, order="F")
        return residual_matrix

    @property
    def factor_residuals_(self):
        """Returns the residual matrix to a vectorised VAR model."""
        X, y = self._build_X_y()

        # factor model coefficients
        var_indices = list(
            np.concatenate(
                [
                    np.arange(
                        1 + self.k_factors,
                        1 + self.k_factors + self.n_series * self.p_lags,
                    )
                    + self.m_features * i
                    for i in range(self.n_series)
                ]
            )
        )
        betas = self.coef_.copy()
        betas[var_indices] = 0

        residuals = y - X.dot(betas.reshape(-1, 1))
        residual_matrix = residuals.reshape(-1, self.n_series, order="F")
        return residual_matrix

    def make_cv_splitter(self, folds=10):
        """Returns a PredefinedSplit object for cross validation."""

        # shapes
        length = self.t_periods // folds
        resid = self.t_periods % folds

        # build single series split
        single_series_split = []
        for i in range(folds - 1, -1, -1):
            single_series_split += length * [i]
            if i < resid:
                single_series_split += [i]

        # make splitter object
        split = self.n_series * single_series_split
        splitter = PredefinedSplit(split)
        return splitter

    def fit_elastic_net_cv(
        self, grid, folds=10, return_cv=False, weighting_method="OLS", **kwargs
    ):
        """Fits the FAVAR coefficients using the glmnet routine.
        Uses OLS inferred penalty weights.
        Sets the model coefficients as the best estimator.
        return_cv can be set to True to return CV object.
        """
        # build inputs
        X, y = self._build_X_y()
        penalty_weights = self._penalty_weights(method=weighting_method)
        split = self.make_cv_splitter(folds=folds)
        elnet = ElasticNet(intercept=False, standardize=False)

        # estimate
        cv = GridSearchCV(elnet, grid, cv=split, n_jobs=-1, verbose=1, **kwargs)
        cv.fit(X, y, penalty_weights=penalty_weights)

        self.coef_ = cv.best_estimator_.coef_.ravel()
        if return_cv:
            return cv

    def residual_cov_(self, method="sample"):
        """Returns the residual covariance.
        Implemented methods are:
        - sample: The sample covariance
        - OAS: Oracle approximation shrinkage
        - LW: Ledoit-Wolf
        - GLASSO: Graphical LASSO, cross-validated
        """
        residuals = self.residuals_

        if method == "sample":
            cov = np.cov(residuals.T)
        elif method == "OAS":
            lw = OAS().fit(residuals)
            cov = np.array(lw.covariance_)
        elif method == "LW":
            lw = LedoitWolf().fit(residuals)
            cov = np.array(lw.covariance_)
        elif method == "GLASSO":
            lw = GraphicalLassoCV().fit(residuals)
            cov = np.array(lw.covariance_)
        else:
            raise NotImplementedError("method not implemented")

        return cov

    def get_lambda_path(self, nlambda=10, **kwargs):
        """Returns the lambda path chosen by the glmnet routine."""
        import glmnet_python
        from glmnet import glmnet

        X, y = self._build_X_y()
        penalty_weights = self._penalty_weights(method="OLS")
        fit = glmnet(
            x=X, y=y, nlambda=nlambda, penalty_factor=penalty_weights, **kwargs
        )
        return fit["lambdau"]
