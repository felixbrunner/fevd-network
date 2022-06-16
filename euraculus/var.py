"""

"""

import copy

import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from euraculus.net import AdaptiveElasticNet, ElasticNet


class VAR:
    """Vector auto-regression.

    Instances of this class handle the transformations necessary to estimate
    VAR parameters and directly invokes the chosen estimation routines.

    Attributes:
        has_intercepts: Indicates if the VAR model includes a constant vector.
        p_lags: The order of lags in the VAR(p) model.
        is_fitted: Indicates if the VAR model has been fitted to data.

    Additional attributes:
        var_matrices_: The VAR coefficient matrices.
        var_1_matrix_: The VAR matrix corresponding to the first lag.
        intercepts_: The model intercepts.

    """

    def __init__(
        self,
        has_intercepts: bool = True,
        p_lags: int = 1,
    ):
        """Initiates the VAR object with descriptive attributes."""
        self.has_intercepts = has_intercepts
        self.p_lags = p_lags
        self.is_fitted = False

    @property
    def var_matrices_(self) -> list:
        """List of matrices of VAR coefficients."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self._var_matrices_

    @property
    def var_1_matrix_(self) -> np.ndarray:
        """Matrix of first lag VAR coefficients."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self._var_matrices_[0]

    @property
    def intercepts_(self) -> np.ndarray:
        """Vector of model intercepts."""
        if not self.has_intercepts:
            raise ValueError("Model does not have intercepts")
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self._intercepts_

    @property
    def _coef_(self):
        """The regression coefficients stacked in a vector."""
        coef_ = np.concatenate([self.intercepts_] + self.var_matrices_, axis=1).ravel()
        return coef_

    @property
    def n_series_(self) -> int:
        """The number of dependent variables in the fitted VAR model."""
        return self.var_1_matrix_.shape[1]

    @property
    def m_features_(self) -> int:
        """The number of features used to explain each dependent variable."""
        return self.has_intercepts + self.n_series_ * self.p_lags

    @property
    def df_used_(self) -> int:
        """The number of degrees of freedom used in the estimation.

        The df_used_ is defined as all nonzero estimates in the intercepts and
        in all VAR coefficient matrices together.

        """
        df_used = np.count_nonzero(self.intercepts_) + np.count_nonzero(
            self.var_matrices_
        )
        return df_used

    @property
    def df_full_(self) -> int:
        """The number of elements in all coeffient matrices.

        The df_full_ is defined as the number of all parameters the model can
        potentially use.

        """
        df_full = np.size(self.intercepts_) + np.size(self.var_matrices_)
        return df_full

    @property
    def coef_density_(self) -> float:
        """The estimate density of a trained model.

        Density is defined as the share of nonzero coefficients.

        """
        density = self.df_used_ / self.df_full_
        return density

    @property
    def coef_sparsity_(self) -> float:
        """The estimate sparsity of a trained model.

        Sparsity is defined as the share of zero coefficients.

        """
        sparsity = 1 - self.coef_density_
        return sparsity

    @property
    def var_density_(self) -> float:
        """The VAR coefficient estimate density of a trained model.

        Density is defined as the share of nonzero coefficients.

        """
        density = np.count_nonzero(self.var_matrices_) / np.size(self.var_matrices_)
        return density

    @property
    def var_sparsity_(self) -> float:
        """The VAR coefficient estimate sparsity of a trained model.

        Sparsity is defined as the share of zero coefficients.

        """
        sparsity = 1 - self.var_density_
        return sparsity

    def _build_y(self, var_data: np.ndarray) -> np.ndarray:
        """Create an array of dependent variables reshaped for estimation.

        Args:
            var_data: (t_periods, n_series) array with observations.

        Returns:
            y: (t_periods*n_series,) array reshaped for regression form.

        """
        y = var_data.values[self.p_lags :].reshape(-1, 1, order="F")
        return y

    def _build_X_block(
        self,
        var_data: np.ndarray,
        add_intercepts: bool,
    ) -> np.ndarray:
        """Create an array of independent data for a single response variable.

        The returned array consists of:
            constant: a vector of ones if the model has intercepts.
            var_data: the lagged observations for VAR coefficient estimation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            add_intercepts: Indicates whether a constant is added.

        Returns:
            X_block: (t_periods, m_features) array for the construction of X.

        """
        # setup
        t_periods = var_data.shape[0] - self.p_lags
        elements = []

        # constant
        if add_intercepts:
            elements += [np.ones([t_periods, 1])]

        # var data lags
        elements += [var_data[self.p_lags - 1 : -1]]
        if self.p_lags > 1:
            for l in range(self.p_lags - 1):
                elements += [var_data[self.p_lags - 2 - l : -2 - l]]

        # build block
        X_block = np.concatenate(elements, axis=1)
        return X_block

    def _build_X(
        self,
        var_data: np.ndarray,
        add_intercepts: bool,
    ) -> sp.sparse.csc.csc_matrix:
        """Create a sparse diagonal block matrix of independent data for estimation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            add_intercepts: Indicates whether a constant is added.

        Returns:
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form.

        """
        # size
        n_series = var_data.shape[1]

        # build
        X_block = self._build_X_block(
            var_data=var_data,
            add_intercepts=add_intercepts,
        )
        X = sp.sparse.kron(sp.sparse.eye(n_series), X_block, format="csc")
        return X

    def _build_X_y(
        self,
        var_data: np.ndarray,
        add_intercepts: bool,
    ) -> tuple:
        """Create data matrices for estimation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            add_intercepts: Indicates whether a constant is added.

        Returns:
            y: (t_periods*n_series,) array reshaped for regression form.
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form.

        """
        X = self._build_X(
            var_data=var_data,
            add_intercepts=add_intercepts,
        )
        y = self._build_y(var_data=var_data)
        return (X, y)

    def _extract_intercepts(
        self,
        coef_: np.ndarray,
        n_series: int,
    ) -> np.ndarray:
        """Return a numpy array of intercepts.

        Args:
            coef_: A vector of fitted regression coefficients.
            n_series: The number of independent variables when fitting.

        Returns:
            intercepts: (n_series, 1) vector of constant terms.

        """
        m_features = self.has_intercepts + n_series * self.p_lags
        indices = list(
            np.concatenate(
                [
                    np.arange(0, 0 + self.has_intercepts) + m_features * i
                    for i in range(n_series)
                ]
            )
        )
        intercepts = coef_[indices].reshape(-1, 1)
        return intercepts

    def _extract_var_matrices(
        self,
        coef_: np.ndarray,
        n_series: int,
    ) -> list:
        """Return a list of numpy arrays with VAR coefficients.

        Args:
            coef_: A vector of fitted regression coefficients.
            n_series: The number of independent variables when fitting.

        Returns:
            var_matrices: List of p_lags (n_series, n_series) matrices with
                VAR coefficient estimates.

        """
        var_matrices = []
        m_features = self.has_intercepts + n_series * self.p_lags
        for l in range(self.p_lags):
            indices = list(
                np.concatenate(
                    [
                        np.arange(self.has_intercepts, self.has_intercepts + n_series)
                        + m_features * i
                        + n_series * l
                        for i in range(n_series)
                    ]
                )
            )
            var_matrix = coef_[indices].reshape(n_series, n_series)
            var_matrices += [var_matrix]
        return var_matrices

    def fit_ols(
        self,
        var_data: np.ndarray,
        return_model: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using OLS.

        Args:
            var_data: (t_periods, n_series) array with observations.
            return_model: Indicates whether to return the fitted model.

        Returns:
            model (optional): The LinearRegression object fitted to the data.

        """
        # build inputs
        X, y = self._build_X_y(var_data=var_data, add_intercepts=self.has_intercepts)
        n_series = var_data.shape[1]

        # estimate
        model = LinearRegression(fit_intercept=False, **kwargs)
        model.fit(X, y)

        # store coefficient estimates
        coef_ = model.coef_.ravel()
        self._intercepts_ = self._extract_intercepts(coef_=coef_, n_series=n_series)
        self._var_matrices_ = self._extract_var_matrices(coef_=coef_, n_series=n_series)
        self.is_fitted = True

        # returns
        if return_model:
            return model

    def _scale_data(
        self,
        data: np.ndarray,
        demean: bool = True,
    ) -> np.ndarray:
        """Scale input data to be centered at zero, with variance one.

        Note that centering is optional.

        Args:
            data: The data array to be standardized, usually the var_data.

        Returns:
            scaled_data: The standardized data array.

        """
        # fit
        levels = data.mean().values
        scales = data.std().values

        # transform
        scaled_data = (data - levels) / scales
        if not demean:
            scaled_data += levels

        return scaled_data

    def _scale_coefs(self, model, var_data: np.ndarray) -> np.ndarray:
        """Recovers the original scale of the coefficeints and adds intercepts.

        After estimating the coefficients on a standardised version of the data,
        coefficients need to be rescaled and intercepts added back in.

        Args:
            model: Model estimated on scaled data without intercepts.
                Has to have a coef_ attribute.
            var_data: (t_periods, n_series) array with observations.

        Returns:
            scaled_coef: The rescaled model coefficent vector.
                Includes intercepts if the original model has intercepts.

        """
        # dimensions
        n_series = var_data.shape[1]
        m_features = n_series * self.p_lags

        # construct inputs
        y_levels = var_data.mean().values.reshape(n_series, 1)
        y_scales = var_data.std().values.reshape(n_series, 1)
        x_block = self._build_X_block(var_data=var_data, add_intercepts=False)
        x_levels = x_block.mean(axis=0).reshape(1, m_features)
        x_scales = x_block.std(axis=0).reshape(1, m_features)

        # rescale coefficients
        coef_block = model.coef_.reshape(n_series, m_features)
        scaled_coef_block = coef_block * y_scales / x_scales

        # add intercepts
        if self.has_intercepts:
            intercepts = y_levels - scaled_coef_block @ x_levels.T
            scaled_coef_block = np.concatenate([intercepts, scaled_coef_block], axis=1)

        # generate output
        scaled_coef = scaled_coef_block.ravel()
        return scaled_coef

    def _make_penalty_weights(
        self,
        n_series: int,
        penalize_diagonals: bool = True,
    ) -> np.ndarray:
        """Create an array that indicates which variables to penalize.

        Serves to exclude diagonal entries in the VAR matrices from penalisation.

        Args:
            n_series: The number of independent variables when fitting.
            penalize_diagonal: Indicates if diagonal VAR entries are to be penalized.

        Returns:
            penalty_weights: Array to indicate which coefficients are to be
                excluded from penalization during estimation.

        """
        penalty_weights = np.ones(shape=(n_series**2 * self.p_lags))
        if not penalize_diagonals:
            indices = list(
                np.concatenate(
                    [
                        np.arange(
                            i + (self.p_lags * n_series) * i,
                            (self.p_lags * n_series) * (i + 1),
                            n_series,
                        )
                        for i in range(n_series)
                    ]
                )
            )
            penalty_weights[indices] = 0
        return penalty_weights

    def _build_inputs(
        self,
        var_data: np.ndarray,
        penalize_diagonals: bool,
    ) -> tuple:
        """Builds the inputs needed to fit a regularized regression.

        Args:
            var_data: (t_periods, n_series) array with observations.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.

        Returns:
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form
            y: (t_periods*n_series,) array reshaped for regression form.
            penalty_weights: Array of ones and zeros to indicate which
                coefficients should be penalized.

        """
        # dimensions
        n_series = var_data.shape[1]

        # scaled data
        scaled_var_data = self._scale_data(
            data=var_data,
            demean=self.has_intercepts,
        )

        # regression inputs
        X, y = self._build_X_y(
            var_data=scaled_var_data,
            add_intercepts=False,
        )
        penalty_weights = self._make_penalty_weights(
            n_series=n_series,
            penalize_diagonals=penalize_diagonals,
        )
        return (X, y, penalty_weights)

    def _store_estimates(
        self,
        var_data: np.ndarray,
        n_series: int,
        model: sklearn.base.BaseEstimator,
    ) -> None:
        """Stores the regression coefficients in the FactorVAR object.

        Args:
            var_data: (t_periods, n_series) array with observations.
            n_series: The number of dependent variables when fitting.
            model: The fitted model which hold a 'coef_' vector.

        """
        # self._coef_ = self._scale_coefs(
        #     model=model,
        #     var_data=var_data,
        # )
        coef_ = self._scale_coefs(
            model=model,
            var_data=var_data,
        )
        self._intercepts_ = self._extract_intercepts(
            coef_=coef_,
            n_series=n_series,
        )
        self._var_matrices_ = self._extract_var_matrices(
            coef_=coef_,
            n_series=n_series,
        )

    def fit_elastic_net(
        self,
        var_data: np.ndarray,
        alpha: float = 0.1,
        lambdau: float = 0.1,
        penalize_diagonals: bool = True,
        return_model: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using the glmnet routine.

        Args:
            var_data: (t_periods, n_series) array with observations.
            alpha: The ratio of L1 penalisation to L2 penalisation, default=0.1.
            lambdau: The penalty factor over all penalty terms, default=0.1.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            return_model: Indicates whether to return the fitted model.

        Returns:
            model (optional): The LinearRegression object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            penalize_diagonals=penalize_diagonals,
        )

        # estimate
        model = ElasticNet(
            alpha=alpha, lambdau=lambdau, intercept=False, standardize=False, **kwargs
        )
        model.fit(
            X,
            y,
            penalty_weights=penalty_weights,
        )

        # store estimates
        self._store_estimates(
            var_data=var_data,
            n_series=n_series,
            model=model,
        )
        self.is_fitted = True

        # returns
        if return_model:
            return model

    def fit_adaptive_elastic_net(
        self,
        var_data: np.ndarray,
        alpha: float = 0.1,
        lambdau: float = 0.1,
        ini_alpha: float = 0.01,
        ini_lambdau: float = None,
        penalize_diagonals: bool = True,
        return_model: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using the adaptive elastic net.

        Args:
            var_data: (t_periods, n_series) array with observations.
            alpha: The ratio of L1 penalisation to L2 penalisation, default=0.1.
            lambdau: The penalty factor over all penalty terms, default=0.1.
            init_alpha: The ratio of L1 to L2 penalisation in the first estimation,
                default=0.01.
            ini_lambdau: The penalty factor in the first estimation, default=None.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            return_model: Indicates whether to return the fitted model.

        Returns:
            model (optional): The LinearRegression object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            penalize_diagonals=penalize_diagonals,
        )

        # estimate
        model = AdaptiveElasticNet(
            alpha=alpha,
            lambdau=lambdau,
            ini_alpha=ini_alpha,
            ini_lambdau=ini_lambdau,
            intercept=False,
            standardize=False,
            **kwargs,
        )
        model.fit(
            X,
            y,
            penalty_weights=penalty_weights,
        )

        # store estimates
        self._store_estimates(
            var_data=var_data,
            n_series=n_series,
            model=model,
        )
        self.is_fitted = True

        # returns
        if return_model:
            return model

    def _make_cv_splitter(
        self,
        var_data: np.ndarray,
        folds: int = 12,
    ) -> sklearn.model_selection._split.PredefinedSplit:
        """Creates a PredefinedSplit object for cross validation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            folds: The number of folds used for cross-validation.

        Returns:
            splitter: Cross-validation sample splits.

        """
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
        self,
        var_data: np.ndarray,
        grid: dict,
        folds: int = 12,
        penalize_diagonals: bool = True,
        return_cv: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using adaptive elastic net with cross-validation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            grid: Hyperparameter grid as dict of iterables.
            folds: The number of folds used for cross-validation.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            return_cv: Indicates whether to return the cross-validation object.

        Returns:
            cv (optional): The GridSearchCV object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            penalize_diagonals=penalize_diagonals,
        )

        # set up CV
        split = self._make_cv_splitter(var_data=var_data, folds=folds)
        elnet = ElasticNet(intercept=False, standardize=False)

        # estimate
        cv = GridSearchCV(
            elnet,
            grid,
            cv=split,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            **kwargs,
        )
        cv.fit(
            X,
            y,
            split=split,
            penalty_weights=penalty_weights,
        )

        # store estimates
        model = cv.best_estimator_
        self._store_estimates(
            var_data=var_data,
            n_series=n_series,
            model=model,
        )
        self.is_fitted = True

        # returns
        if return_cv:
            return cv

    def fit_adaptive_elastic_net_cv(
        self,
        var_data: np.ndarray,
        grid: dict,
        folds: int = 12,
        penalize_diagonals: bool = True,
        return_cv: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using the glmnet routine with cross-validation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            grid: Hyperparameter grid as dict of iterables.
            folds: The number of folds used for cross-validation.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            return_cv: Indicates whether to return the cross-validation object.

        Returns:
            cv (optional): The GridSearchCV object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            penalize_diagonals=penalize_diagonals,
        )

        # set up CV
        split = self._make_cv_splitter(var_data=var_data, folds=folds)
        elnet = AdaptiveElasticNet(intercept=False, standardize=False)
        elnet.fit(
            X,
            y,
            ini_split=split,
            penalty_weights=penalty_weights,
        )  # required to update the penalty weights only once

        # estimate
        cv = GridSearchCV(
            elnet,
            grid,
            cv=split,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            **kwargs,
        )
        cv.fit(
            X,
            y,
            split=split,
            penalty_weights=penalty_weights,
        )

        # store estimates
        model = cv.best_estimator_
        self._store_estimates(
            var_data=var_data,
            n_series=n_series,
            model=model,
        )
        self.is_fitted = True

        # returns
        if return_cv:
            return cv

    def fit(
        self,
        method: str = "OLS",
        **kwargs,
    ) -> None:
        """Fit the VAR model through one of the available fit methods.

        Available fitting methods are:
            OLS
            ElasticNet
            AdaptiveElasticNet
            ElasticNetCV
            AdaptiveElasticNetCV

        Args:
            method: Indicates the fitting method to use.

        """
        if method == "OLS":
            self.fit_ols(**kwargs)
        elif method == "ElasticNet":
            self.fit_elastic_net(**kwargs)
        elif method == "AdaptiveElasticNet":
            self.fit_adaptive_elastic_net(**kwargs)
        elif method == "ElasticNetCV":
            self.fit_elastic_net_cv(**kwargs)
        elif method == "AdaptiveElasticNetCV":
            self.fit_adaptive_elastic_net_cv(**kwargs)
        else:
            raise Exception("Fit method not implemented.")

    def predict(self, var_data: np.ndarray, **kwargs) -> pd.DataFrame:
        """Calculate fitted values for the fitted model and input data.

        Args:
            var_data: (t_periods, n_series) array with observations.

        Returns:
            y_pred: (t_periods-p_lags, n_series) dataframe with fitted values.

        """
        # setup
        t_periods = var_data.shape[0] - self.p_lags
        n_series = var_data.shape[1]
        X = self._build_X(
            var_data=var_data, add_intercepts=self.has_intercepts, **kwargs
        )

        # make predictions
        y_pred = (X @ self._coef_).reshape(t_periods, n_series, order="F")
        if type(var_data) == pd.DataFrame:
            y_pred = pd.DataFrame(
                y_pred, index=var_data.index[self.p_lags :], columns=var_data.columns
            )
        return y_pred

    def residuals(self, var_data: np.ndarray, **kwargs) -> pd.DataFrame:
        """Calculate prediction residuals for the fitted model and input data.

        Args:
            var_data: (t_periods, n_series) array with observations.

        Returns:
            residuals: (t_periods-p_lags, n_series) dataframe with residuals.

        """
        y_pred = self.predict(var_data=var_data, **kwargs)
        residuals = var_data[self.p_lags :] - y_pred
        return residuals

    def r2(
        self,
        var_data: np.ndarray,
        weighting: str = "equal",
        **kwargs,
    ) -> float:
        """Calculate goodness of fit for the fitted model and input data.

        Args:
            var_data: (t_periods, n_series) array with observations.
            weighting: Indicates how to weigh dependent variables. Available
                options are ["equal", "variance", "granular"].

        Returns:
            r2: The goodness of fit statistic for the input data and the model.

        """
        levels = var_data[self.p_lags :].mean().values
        tss = np.nansum((var_data[self.p_lags :] - levels) ** 2, axis=0)
        rss = np.nansum(self.residuals(var_data, **kwargs) ** 2, axis=0)
        if weighting == "equal":
            r2 = 1 - rss.sum() / tss.sum()
        elif weighting == "variance":
            variances = var_data[self.p_lags :].mean().values
            r2 = 1 - (rss / variances).sum() / (tss / variances).sum()
        elif weighting == "granular":
            r2 = 1 - rss / tss
        else:
            raise ValueError("weighting method '{}' not available".format(weighting))
        return r2

    def copy(self):
        """Returns a copy of the VAR object."""
        return copy.deepcopy(self)


class FactorVAR(VAR):
    """Vector auto-regression with common factors.

    Instances of this class handle the transformations necessary to estimate
    VAR and factor loading parameters and directly invoke the chosen
    estimation routines.

    Attributes:
        has_intercepts: Indicates if the VAR model includes a constant vector.
        p_lags: The order of lags in the VAR(p) model.
        is_fitted: Indicates if the VAR model has been fitted to data.

    Additional attributes:
        var_matrices_:The VAR coefficient matrices.
        var_1_matrix_: The VAR matrix corresponding to the first lag.
        intercepts_: The model intercepts.
        factor_loadings_: The model factor loadings.
        k_factors_: The number of factors in the model.

    """

    def __init__(
        self,
        has_intercepts: bool = True,
        p_lags: int = 1,
    ):
        """Initiates the VAR object with descriptive attributes."""
        VAR.__init__(self, has_intercepts=has_intercepts, p_lags=p_lags)

    @property
    def factor_loadings_(self) -> np.ndarray:
        """Matrix with factor loadings."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self._factor_loadings_

    @property
    def _coef_(self):
        """The regression coefficients stacked in a vector."""
        coef_ = np.concatenate(
            [self.intercepts_, self.factor_loadings_] + self.var_matrices_, axis=1
        ).ravel()
        return coef_

    @property
    def k_factors_(self) -> int:
        """The number of common factors in the fitted FactorVAR model."""
        return self.factor_loadings_.shape[1]

    @property
    def m_features_(self) -> int:
        """The number of features used to explain each dependent variable."""
        return self.has_intercepts + self.k_factors_ + self.n_series_ * self.p_lags

    @property
    def df_used_(self) -> int:
        """The number of degrees of freedom used in the estimation.

        The df_used_ is defined as all nonzero estimates in the intercepts,
        factor loadings and all VAR coefficient matrices together.

        """
        df_used = (
            np.count_nonzero(self.intercepts_)
            + np.count_nonzero(self.factor_loadings_)
            + np.count_nonzero(self.var_matrices_)
        )
        return df_used

    @property
    def df_full_(self) -> int:
        """The number of elements in all coeffient matrices.

        The df_full_ is defined as the number of all parameters the model can
        potentially use.

        """
        df_full = (
            np.size(self.intercepts_)
            + np.size(self.factor_loadings_)
            + np.size(self.var_matrices_)
        )
        return df_full

    @property
    def factor_loading_density_(self) -> float:
        """The factor loading estimate density of a trained model.

        Density is defined as the share of nonzero coefficients.

        """
        density = np.count_nonzero(self.factor_loadings_) / np.size(
            self.factor_loadings_
        )
        return density

    @property
    def factor_loading_sparsity_(self) -> float:
        """The factor loading estimate sparsity of a trained model.

        Sparsity is defined as the share of zero coefficients.

        """
        sparsity = 1 - self.factor_loading_density_
        return sparsity

    def _build_X_block(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray,
        add_intercepts: bool,
    ) -> np.ndarray:
        """Create an array of independent data for a single response variable.

        The returned array consists of:
            constant: A vector of ones if the model has intercepts.
            factor_data: The contemporaneous common factor observations.
            var_data: The lagged observations for VAR coefficient estimation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            add_intercepts: Indicates whether a constant is added.

        Returns:
            X_block: (t_periods, m_features) array for the construction of X.

        """
        # setup
        t_periods = var_data.shape[0] - self.p_lags
        elements = []

        # constant
        if add_intercepts:
            elements += [np.ones([t_periods, 1])]

        # exogenous regressors
        if factor_data is not None:
            if not factor_data.shape[0] == var_data.shape[0]:
                raise ValueError("number of observations unequal")
            elements += [factor_data[self.p_lags :]]

        # var data lags
        elements += [var_data[self.p_lags - 1 : -1]]
        if self.p_lags > 1:
            for l in range(self.p_lags - 1):
                elements += [var_data[self.p_lags - 2 - l : -2 - l]]

        # build block
        X_block = np.concatenate(elements, axis=1)
        return X_block

    def _build_X(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray,
        add_intercepts: bool,
    ) -> sp.sparse.csc.csc_matrix:
        """Create a sparse diagonal block matrix of independent data for estimation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            add_intercepts: Indicates whether a constant is added.

        Returns:
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form.

        """
        # size
        n_series = var_data.shape[1]

        # build
        X_block = self._build_X_block(
            var_data=var_data,
            factor_data=factor_data,
            add_intercepts=add_intercepts,
        )
        X = sp.sparse.kron(sp.sparse.eye(n_series), X_block, format="csc")
        return X

    def _build_X_y(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray,
        add_intercepts: bool,
    ) -> tuple:
        """Create data matrices for estimation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            add_intercepts: Indicates whether a constant is added.

        Returns:
            y: (t_periods*n_series,) array reshaped for regression form.
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form.

        """
        X = self._build_X(
            var_data=var_data,
            factor_data=factor_data,
            add_intercepts=add_intercepts,
        )
        y = self._build_y(var_data=var_data)
        return (X, y)

    def _extract_intercepts(
        self,
        coef_: np.ndarray,
        n_series: int,
        k_factors: int,
    ) -> np.ndarray:
        """Return a numpy array of intercepts.

        Args:
            coef_: A vector of fitted regression coefficients.
            n_series: The number of independent variables when fitting.
            k_factors: The number of common factors when fitting.

        Returns:
            intercepts: (n_series, 1) vector of constant terms.

        """
        m_features = self.has_intercepts + k_factors + n_series * self.p_lags
        indices = list(
            np.concatenate(
                [
                    np.arange(0, 0 + self.has_intercepts) + m_features * i
                    for i in range(n_series)
                ]
            )
        )
        intercepts = coef_[indices].reshape(-1, 1)
        return intercepts

    def _extract_factor_loadings(
        self,
        coef_: np.ndarray,
        n_series: int,
        k_factors: int,
    ) -> np.ndarray:
        """Return a numpy array of factor loadings.

        Args:
            coef_: A vector of fitted regression coefficients.
            n_series: The number of independent variables when fitting.
            k_factors: The number of common factors when fitting.

        Returns:
            factor_loadings: (n_series, k_factors) matrix of factor loadings.

        """
        m_features = self.has_intercepts + k_factors + n_series * self.p_lags
        indices = list(
            np.concatenate(
                [
                    np.arange(self.has_intercepts, self.has_intercepts + k_factors)
                    + m_features * i
                    for i in range(n_series)
                ]
            )
        )
        factor_loadings = coef_[indices].reshape(-1, k_factors)
        return factor_loadings

    def _extract_var_matrices(
        self,
        coef_: np.ndarray,
        n_series: int,
        k_factors: int,
    ) -> np.ndarray:
        """Return a list of numpy arrays with VAR coefficients.

        Args:
            coef_: A vector of fitted regression coefficients.
            n_series: The number of independent variables when fitting.
            k_factors: The number of common factors when fitting.

        Returns:
            var_matrices: List of p_lags (n_series, n_series) matrices with
                VAR coefficient estimates.

        """
        var_matrices = []
        m_features = self.has_intercepts + k_factors + n_series * self.p_lags
        for l in range(self.p_lags):
            indices = list(
                np.concatenate(
                    [
                        np.arange(
                            self.has_intercepts + k_factors,
                            self.has_intercepts + k_factors + n_series,
                        )
                        + m_features * i
                        + n_series * l
                        for i in range(n_series)
                    ]
                )
            )
            var_matrix = coef_[indices].reshape(n_series, n_series)
            var_matrices += [var_matrix]
        return var_matrices

    def fit_ols(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray = None,
        return_model: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using OLS.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            return_model: Indicates whether to return the fitted model.

        Returns:
            model (optional): The LinearRegression object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        k_factors = factor_data.shape[1]
        X, y = self._build_X_y(
            var_data=var_data,
            factor_data=factor_data,
            add_intercepts=self.has_intercepts,
        )

        # estimate
        model = LinearRegression(fit_intercept=False, **kwargs)
        model.fit(X, y)

        # store coefficient estimates
        coef_ = model.coef_.ravel()
        self._intercepts_ = self._extract_intercepts(
            coef_=coef_, n_series=n_series, k_factors=k_factors
        )
        self._factor_loadings_ = self._extract_factor_loadings(
            coef_=coef_, n_series=n_series, k_factors=k_factors
        )
        self._var_matrices_ = self._extract_var_matrices(
            coef_=coef_, n_series=n_series, k_factors=k_factors
        )
        self.is_fitted = True

        # returns
        if return_model:
            return model

    def _scale_coefs(
        self,
        model,
        var_data: np.ndarray,
        factor_data: np.ndarray,
    ) -> np.ndarray:
        """Recovers the original scale of the coefficients and adds intercepts.

        After estimating the coefficients on a standardised version of the data,
        coefficients need to be rescaled and intercepts added back in.

        Args:
            model: Model estimated on scaled data without intercepts.
                Has to have a coef_ attribute.
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            scaled_coef: The rescaled model coefficent vector.
                Includes intercepts if the original model has intercepts.

        """
        # dimensions
        n_series = var_data.shape[1]
        k_factors = factor_data.shape[1]
        m_features = k_factors + n_series * self.p_lags

        # construct inputs
        y_levels = var_data.mean().values.reshape(n_series, 1)
        y_scales = var_data.std().values.reshape(n_series, 1)
        x_block = self._build_X_block(
            var_data=var_data,
            factor_data=factor_data,
            add_intercepts=False,
        )
        x_levels = x_block.mean(axis=0).reshape(1, m_features)
        x_scales = x_block.std(axis=0).reshape(1, m_features)

        # rescale coefficients
        coef_block = model.coef_.reshape(n_series, m_features)
        scaled_coef_block = coef_block * y_scales / x_scales

        # add intercepts
        if self.has_intercepts:
            intercepts = y_levels - scaled_coef_block @ x_levels.T
            scaled_coef_block = np.concatenate([intercepts, scaled_coef_block], axis=1)

        # generate output
        scaled_coef = scaled_coef_block.ravel()
        return scaled_coef

    def _make_penalty_weights(
        self,
        n_series: int,
        k_factors: int,
        penalize_diagonals: bool = True,
        penalize_factors: bool = True,
    ) -> np.ndarray:
        """Create an array that indicates which variables to penalize.

        Serves to exclude diagonal entries in the VAR matrices and factor
        loadings from penalisation.

        Args:
            n_series: The number of independent variables when fitting.
            penalize_diagonal: Indicates if diagonal VAR entries are to be penalized.
            penalize_factors: Indicates if factor loadings are to be penalized.

        Returns:
            penalty_weights: Array of ones and zeros to indicate which
                coefficients should be penalized.

        """
        penalty_weights = np.ones(shape=((k_factors + n_series) * n_series))
        if not penalize_diagonals:
            indices = list(
                np.concatenate(
                    [
                        np.arange(
                            k_factors + i + (k_factors + n_series * self.p_lags) * i,
                            (k_factors + n_series * self.p_lags) * (i + 1),
                            n_series,
                        )
                        for i in range(n_series)
                    ]
                )
            )
            penalty_weights[indices] = 0
        if not penalize_factors:
            indices = list(
                np.concatenate(
                    [
                        np.arange(
                            0 + i * (k_factors + n_series * self.p_lags),
                            k_factors + i * (k_factors + n_series * self.p_lags),
                        )
                        for i in range(n_series)
                    ]
                )
            )
            penalty_weights[indices] = 0
        return penalty_weights

    def _build_inputs(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray,
        penalize_diagonals: bool,
        penalize_factors: bool,
    ) -> tuple:
        """Builds the inputs needed to fit a regularized regression.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            penalize_factors: Indicates if factor loadings are to be penalized.

        Returns:
            X: (t_periods*n_series, m_features*n_series) array reshaped
                for regression form
            y: (t_periods*n_series,) array reshaped for regression form.
            penalty_weights: Array of ones and zeros to indicate which
                coefficients should be penalized.

        """
        # dimensions
        n_series = var_data.shape[1]
        k_factors = factor_data.shape[1]

        # scaled data
        scaled_var_data = self._scale_data(
            data=var_data,
            demean=self.has_intercepts,
        )
        scaled_factor_data = self._scale_data(
            data=factor_data,
            demean=self.has_intercepts,
        )

        # regression inputs
        X, y = self._build_X_y(
            var_data=scaled_var_data,
            factor_data=scaled_factor_data,
            add_intercepts=False,
        )
        penalty_weights = self._make_penalty_weights(
            n_series=n_series,
            k_factors=k_factors,
            penalize_diagonals=penalize_diagonals,
            penalize_factors=penalize_factors,
        )
        return (X, y, penalty_weights)

    def _store_estimates(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray,
        n_series: int,
        k_factors: int,
        model: sklearn.base.BaseEstimator,
    ) -> None:
        """Stores the regression coefficients in the FactorVAR object.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            n_series: The number of dependent variables when fitting.
            k_factors: The number of factors when fitting.
            model: The fitted model which hold a 'coef_' vector.

        """
        coef_ = self._scale_coefs(
            model=model,
            var_data=var_data,
            factor_data=factor_data,
        )
        self._intercepts_ = self._extract_intercepts(
            coef_=coef_, n_series=n_series, k_factors=k_factors
        )
        self._factor_loadings_ = self._extract_factor_loadings(
            coef_=coef_, n_series=n_series, k_factors=k_factors
        )
        self._var_matrices_ = self._extract_var_matrices(
            coef_=coef_, n_series=n_series, k_factors=k_factors
        )

    def fit_elastic_net(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray = None,
        alpha: float = 0.1,
        lambdau: float = 0.1,
        penalize_diagonals: bool = True,
        penalize_factors: bool = True,
        return_model: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using the glmnet routine.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            alpha: The ratio of L1 penalisation to L2 penalisation, default=0.1.
            lambdau: The penalty factor over all penalty terms, default=0.1.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            penalize_factors: Indicates if factor loadings are to be penalized.
            return_model: Indicates whether to return the fitted model.

        Returns:
            model (optional): The LinearRegression object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        k_factors = factor_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            factor_data=factor_data,
            penalize_diagonals=penalize_diagonals,
            penalize_factors=penalize_factors,
        )

        # estimate
        model = ElasticNet(
            alpha=alpha,
            lambdau=lambdau,
            intercept=False,
            standardize=False,
            **kwargs,
        )
        model.fit(
            X,
            y,
            penalty_weights=penalty_weights,
        )

        # store estimates
        self._store_estimates(var_data, factor_data, n_series, k_factors, model)
        self.is_fitted = True

        # returns
        if return_model:
            return model

    def fit_adaptive_elastic_net(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray = None,
        alpha: float = 0.1,
        lambdau: float = 0.1,
        ini_alpha: float = 0.01,
        ini_lambdau: float = None,
        penalize_diagonals: bool = True,
        penalize_factors: bool = True,
        return_model: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using the adaptive elastic net.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            alpha: The ratio of L1 penalisation to L2 penalisation, default=0.1.
            lambdau: The penalty factor over all penalty terms, default=0.1.
            init_alpha: The ratio of L1 to L2 penalisation in the first estimation,
                default=0.01.
            ini_lambdau: The penalty factor in the first estimation, default=None.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            penalize_factors: Indicates if factor loadings are to be penalized.
            return_model: Indicates whether to return the fitted model.

        Returns:
            model (optional): The LinearRegression object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        k_factors = factor_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            factor_data=factor_data,
            penalize_diagonals=penalize_diagonals,
            penalize_factors=penalize_factors,
        )

        # estimate
        model = AdaptiveElasticNet(
            alpha=alpha,
            lambdau=lambdau,
            ini_alpha=ini_alpha,
            ini_lambdau=ini_lambdau,
            intercept=False,
            standardize=False,
            **kwargs,
        )
        model.fit(
            X,
            y,
            penalty_weights=penalty_weights,
        )

        # store estimates
        self._store_estimates(var_data, factor_data, n_series, k_factors, model)
        self.is_fitted = True

        # returns
        if return_model:
            return model

    def fit_elastic_net_cv(
        self,
        var_data: np.ndarray,
        grid: dict,
        folds: int = 12,
        factor_data: np.ndarray = None,
        penalize_diagonals: bool = True,
        penalize_factors: bool = True,
        return_cv: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using elastic net with cross-validation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            grid: Hyperparameter grid as dict of iterables.
            folds: The number of folds used for cross-validation.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            penalize_factors: Indicates if factor loadings are to be penalized.
            return_cv: Indicates whether to return the cross-validation object.

        Returns:
            cv (optional): The GridSearchCV object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        k_factors = factor_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            factor_data=factor_data,
            penalize_diagonals=penalize_diagonals,
            penalize_factors=penalize_factors,
        )

        # set up CV
        split = self._make_cv_splitter(var_data=var_data, folds=folds)
        elnet = ElasticNet(intercept=False, standardize=False)

        # estimate
        cv = GridSearchCV(
            elnet,
            grid,
            cv=split,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            **kwargs,
        )
        cv.fit(
            X,
            y,
            split=split,
            penalty_weights=penalty_weights,
        )

        # store estimates
        model = cv.best_estimator_
        self._store_estimates(var_data, factor_data, n_series, k_factors, model)
        self.is_fitted = True

        # returns
        if return_cv:
            return cv

    def fit_adaptive_elastic_net_cv(
        self,
        var_data: np.ndarray,
        grid: dict,
        folds: int = 12,
        factor_data: np.ndarray = None,
        penalize_diagonals: bool = True,
        penalize_factors: bool = True,
        return_cv: bool = False,
        **kwargs,
    ) -> None:
        """Fits the VAR coefficients using adaptive elastic net with cross-validation.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            grid: Hyperparameter grid as dict of iterables.
            folds: The number of folds used for cross-validation.
            penalize_diagonals: Indicates if diagonal VAR entries are to be penalized.
            penalize_factors: Indicates if factor loadings are to be penalized.
            return_cv: Indicates whether to return the cross-validation object.

        Returns:
            cv (optional): The GridSearchCV object fitted to the data.

        """
        # build inputs
        n_series = var_data.shape[1]
        k_factors = factor_data.shape[1]
        X, y, penalty_weights = self._build_inputs(
            var_data=var_data,
            factor_data=factor_data,
            penalize_diagonals=penalize_diagonals,
            penalize_factors=penalize_factors,
        )

        # set up CV
        split = self._make_cv_splitter(var_data=var_data, folds=folds)
        elnet = AdaptiveElasticNet(intercept=False, standardize=False)
        elnet.fit(
            X,
            y,
            ini_split=split,
            penalty_weights=penalty_weights,
        )  # required to update the penalty weights only once

        # estimate
        cv = GridSearchCV(
            elnet,
            grid,
            cv=split,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            **kwargs,
        )
        cv.fit(
            X,
            y,
            split=split,
            penalty_weights=penalty_weights,
        )

        # store estimates
        model = cv.best_estimator_
        self._store_estimates(var_data, factor_data, n_series, k_factors, model)
        self.is_fitted = True

        # returns
        if return_cv:
            return cv

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
        factor_predictions = factor_data[self.p_lags :] @ self.factor_loadings_.T
        if self.has_intercepts:
            factor_predictions += self.intercepts_.T
        return factor_predictions

    def factor_residuals(
        self,
        var_data: np.ndarray = None,
        factor_data: np.ndarray = None,
    ):
        """Calculate prediction residuals from the fitted factor loadings and input data.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.

        Returns:
            residuals: (t_periods, n_series) dataframe with factor residuals.

        """
        factor_predictions = self.factor_predict(factor_data=factor_data)
        if type(var_data) == pd.DataFrame:
            residuals = pd.DataFrame(
                var_data[self.p_lags :].values - factor_predictions.values,
                index=var_data[self.p_lags :].index,
                columns=var_data.columns,
            )
        return residuals

    def factor_r2(
        self,
        var_data: np.ndarray = None,
        factor_data: np.ndarray = None,
        weighting: str = "equal",
    ):
        """Calculate goodness of fit from the fitted factor loadings and input data.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            weighting: Indicates how to weigh dependent variables. Available
                options are ["equal", "variance", "granular"].

        Returns:
            factor_r2: The goodness of fit statistic for the factor loadings.

        """
        levels = var_data[self.p_lags :].mean().values
        tss = np.nansum(
            (var_data[self.p_lags :] - levels) ** 2,
            axis=0,
        )
        residuals = self.factor_residuals(var_data=var_data, factor_data=factor_data)
        residuals -= residuals.mean()
        rss = np.nansum(
            residuals**2,
            axis=0,
        )
        if weighting == "equal":
            factor_r2 = 1 - rss.sum() / tss.sum()
        elif weighting == "variance":
            variances = var_data[self.p_lags :].mean().values
            factor_r2 = 1 - (rss / variances).sum() / (tss / variances).sum()
        elif weighting == "granular":
            factor_r2 = 1 - rss / tss
        else:
            raise ValueError("weighting method '{}' not available".format(weighting))
        return factor_r2

    def partial_r2s(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray,
        weighting: str = "equal",
    ) -> dict:
        """Calculate partial goodness of fit for each factor, all factors, and the VAR.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            weighting: Indicates how to weigh dependent variables. Available
                options are ["equal", "variance", "granular"].

        Returns:
            partial_r2s: The partial goodness of fit statistics.

        """

        # full model
        residuals = self.residuals(var_data=var_data, factor_data=factor_data)
        ss_full = np.nansum((residuals - residuals.mean()) ** 2, axis=0)

        # single factor restricted models
        ss_partial = []
        for i_factor in range(factor_data.shape[1]):
            restricted_model = self.copy()
            restricted_model._factor_loadings_[:, i_factor] = 0
            residuals = restricted_model.residuals(
                var_data=var_data, factor_data=factor_data
            )
            ss_partial += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # no factors model
        restricted_model = self.copy()
        restricted_model._factor_loadings_[:] = 0
        residuals = restricted_model.residuals(
            var_data=var_data, factor_data=factor_data
        )
        ss_partial += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # no var model
        restricted_model = self.copy()
        restricted_model._var_matrices_ = [
            np.zeros(m.shape) for m in restricted_model._var_matrices_
        ]
        residuals = restricted_model.residuals(
            var_data=var_data, factor_data=factor_data
        )
        ss_partial += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # calculate partial r2
        partial_r2s = []
        if weighting == "equal":
            for ss_restricted in ss_partial:
                partial_r2s += [1 - ss_full.sum() / ss_restricted.sum()]
        elif weighting == "variance":
            for ss_restricted in ss_partial:
                variances = var_data[self.p_lags :].mean().values
                partial_r2s += [
                    1 - (ss_full / variances).sum() / (ss_restricted / variances).sum()
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
        keys += ["factors", "var"]
        partial_r2s = {k: v for (k, v) in zip(keys, partial_r2s)}
        return partial_r2s

    def component_r2s(
        self,
        var_data: np.ndarray,
        factor_data: np.ndarray,
        weighting: str = "equal",
    ) -> dict:
        """Calculate goodness of fit for each factor, all factors, and the VAR.

        Args:
            var_data: (t_periods, n_series) array with observations.
            factor_data: (t_periods, k_factors) array with factor observations.
            weighting: Indicates how to weigh dependent variables. Available
                options are ["equal", "variance", "granular"].

        Returns:
            partial_r2s: The partial goodness of fit statistics.

        """

        # total variation
        levels = var_data[self.p_lags :].mean().values
        tss = np.nansum(
            (var_data[self.p_lags :] - levels) ** 2,
            axis=0,
        )

        # single factor models
        ss_component = []
        for i_factor in range(factor_data.shape[1]):
            component_model = self.copy()
            component_model._factor_loadings_[:] = 0
            component_model._var_matrices_ = [
                np.zeros(m.shape) for m in component_model._var_matrices_
            ]
            component_model._factor_loadings_[:, i_factor] = self._factor_loadings_[
                :, i_factor
            ]
            residuals = component_model.residuals(
                var_data=var_data, factor_data=factor_data
            )
            ss_component += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # pure factor model
        component_model = self.copy()
        component_model._var_matrices_ = [
            np.zeros(m.shape) for m in component_model._var_matrices_
        ]
        residuals = component_model.residuals(
            var_data=var_data, factor_data=factor_data
        )
        ss_component += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # pure var model
        component_model = self.copy()
        component_model._factor_loadings_[:] = 0
        residuals = component_model.residuals(
            var_data=var_data, factor_data=factor_data
        )
        ss_component += [np.nansum((residuals - residuals.mean()) ** 2, axis=0)]

        # calculate partial r2
        component_r2s = []
        if weighting == "equal":
            for ss_restricted in ss_component:
                component_r2s += [1 - ss_restricted.sum() / tss.sum()]
        elif weighting == "variance":
            for ss_restricted in ss_component:
                variances = var_data[self.p_lags :].mean().values
                component_r2s += [
                    1 - (ss_restricted / variances).sum() / (tss / variances).sum()
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
        keys += ["factors", "var"]
        component_r2s = {k: v for (k, v) in zip(keys, component_r2s)}
        return component_r2s
