import copy

import glmnet_python
import numpy as np
import scipy as sp

from glmnet import glmnet
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ElasticNet(BaseEstimator):
    """Elastic Net estimator based on the Fortran routine glmnet.

    Attributes:
        lambdau: The penalty factor over all penalty terms, default=0.1.
        alpha: The ratio of L1 penalisation to L2 penalisation, default=0.1.
        standardize: Indicates whether features should be standardized
            before estimation, default=False.
        intercept: Indicates whether to include an intercept, default=False.
        threshold: Optimisation convergence threshold, defaut=1e-4.
        max_iter: Maximum number of iterations, default=1e5.

    Examples:
        >>> from src.net import ElasticNet
        >>> import numpy as np
        >>> X = np.arange(100).reshape(100, 1)
        >>> y = np.zeros((100, ))
        >>> net = ElasticNet()
        >>> net.fit(X, y)
        ElasticNet(lambdau=0.1,
                    alpha=0.1,
                    standardize=False,
                    intercept=False,
                    threshold=1e-4,
                    max_iter=1e5)

    """

    def __init__(
        self,
        lambdau: float = 0.1,
        alpha: float = 0.1,
        standardize: bool = False,
        intercept: bool = False,
        threshold: float = 1e-4,
        max_iter: float = 1e5,
        **kwargs,
    ) -> None:
        """Initializes the ElasticNet BaseEstimator object with hyperparameters."""

        self.lambdau = lambdau
        self.alpha = alpha
        self.standardize = standardize
        self.intercept = intercept
        self.threshold = threshold
        self.max_iter = max_iter

        self.__dict__.update(kwargs)

    def _fix_data(self, data: np.ndarray) -> np.ndarray:
        """Sets appropriate types for data inputs.

        Datatypes will be converted to floating point numbers and sparse matrices
        will use the CSC format the glmnet routine knows how to deal with.

        Args:
            data: The unformatted data array.

        Returns:
            data_: The formatted data array.

        """
        data_ = data.astype("float64")
        if sp.sparse.issparse(data):
            data_ = data_.tocsc()
        return data_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty_weights: np.ndarray = None,
        return_fit: bool = False,
        **kwargs,
    ):
        """Fits the model parameters to input data.

        Args:
            X: The training input samples of shape (t_samples, k_features).
                Can be a sparse matrix.
            y: The target values as real numbers of shape
                (n_samples,) or (n_samples, 1).
            penalty_weights: Coefficient penalty weights, zero if not penalised,
                of shape (k_features,), default=None.
            return_fit: Indicates whether full fit statistics are returned
                instead of fitting model inplace, default=False.

        Returns:
            self : The fitted ElasticNet object.
            fit (dict): The glmnet results as obtained from the glmnet routine.

        """
        # dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError("data dimension mismatch")
        k_feat = X.shape[1]

        # set penalty weights
        if penalty_weights is None:
            penalty_weights = np.ones([k_feat])

        # transform inputs for glmnet
        lambdau = np.array([self.lambdau])
        X = self._fix_data(X)
        y = self._fix_data(y)

        # estimate
        fit = glmnet(
            x=X,
            y=y,
            alpha=self.alpha,  # corresponds to kappa hyperparameter
            standardize=self.standardize,  # standardise data before optimisation
            lambdau=lambdau,  # lambda hyperparameter
            penalty_factor=penalty_weights,  # coefficient penalty weight
            intr=self.intercept,  # intercept
            thresh=self.threshold,  # convergence threshold
            maxit=self.max_iter,  # maximum number of iterations
        )

        if return_fit:
            return fit
        else:
            # store results
            self.coef_ = fit["beta"]
            self.R2_ = fit["dev"][0] / 100
            self.df_used_ = fit["df"][0]
            self.is_fitted_ = True
            return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts values for y given X and the parameters of the estimator.

        Args:
            X: The input samples of shape (n_samples, k_features),
                can be a sparse matrix.

        Returns:
            y: Predicted values for the inputs X of shape (n_samples,).

        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        y = X @ self.coef_
        return y

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scorer=None,
    ) -> float:
        """Calculate the score for input data X and labels y.

        The default metric used is the negative of the MSE, but other metrics
        can be provided as a function.

        Args:
            X: The input samples of shape (n_samples, k_features),
                can be a sparse matrix.
            y: Labels of shape (n_samples,) corresponding to the inputs X.
            scorer (optional, function): Alternative scoring function instead of MSE.

        Returns:
            score: The calculated score for the model predictions.

        """
        predictions = self.predict(X)
        if scorer is None:
            score = -mean_squared_error(predictions, y)
        else:
            score = -scorer(predictions, y)
        return score

    def copy(self):
        """Returns a copy of the ElasticNet BaseEstimator."""
        return copy.deepcopy(self)


#     def find_lambda_path(self, X, y, alpha=None, penalty_weights=None, nlambda=10, **kwargs):
#         '''Uses glmnet API to find a plausible path for lambda.
#         '''
#         # dimensions
#         assert X.shape[0] == y.shape[0], \
#             'data dimension mismatch'
#
#         n_obs = y.shape[0]
#         n_feat = X.shape[1]
#
#         # set penalty weights
#         if penalty_weights is None:
#             penalty_weights = np.ones([n_feat])
#
#         # set alpha
#         if alpha is None:
#             alpha = self.alpha
#
#         # transform inputs for glmnet
#         X = self._fix_data(X)
#         y = self._fix_data(y)
#
#         # estimate
#         fit = glmnet(x=X,
#                      y=y,
#                      alpha=alpha,                    # corresponds to kappa hyperparameter
#                      nlambda=nlambda,                # maximum size of lambda search grid
#                      standardize=self.standardize,   # standardise data before optimisation
#                      penalty_factor=penalty_weights, # coefficient penalty weight
#                      intr=self.intercept,            # intercept
#                      thresh=self.threshold,          # convergence threshold
#                      maxit=self.max_iter,            # maximum number of iterations
#                     )
#         lambda_path = fit['lambdau']
#         return lambda_path


#     def find_lambda(self, X, y, lambda_grid=[1], alpha=None, penalty_weights=None, split=None, **kwargs):
#         '''Finds the best lambda among a list of alternatives.
#         Uses 2-fold cross-validation with alternating sampling by default.
#         '''
#         # set up net
#         elnet = self.copy()
#         if alpha:
#             elnet.alpha = alpha
#
#         # set grid
#         if type(lambda_grid) == np.ndarray:
#             lambda_grid = lambda_grid.tolist()
#         grid = {'lambdau': lambda_grid}
#
#         # set split
#         if split is None:
#             split = PredefinedSplit([0, 1]*(len(y)//2))
#         cv = GridSearchCV(elnet, grid, cv=split, **kwargs)
#
#         # fit
#         cv.fit(X, y, penalty_weights=penalty_weights, refit=False)
#
#         # extract
#         lambdau = grid['lambdau'][cv.best_index_]
#
#         return lambdau


class AdaptiveElasticNet(ElasticNet):
    """Implementation of the Adaptive Elastic Net Estimator of Zou/Zhang 2009.

    Attributes:
        gamma: The exponent used to scale the penalty weights, default=1.
        init_alpha: The ratio of L1 to L2 penalisation in the first estimation.
        ini_lambdau: The penalty factor in the first estimation.
        penalty_weights: Coefficient penalty weights, zero if not penalised,
                of shape (k_features,), default=None.

    Attributes inherited from ElasticNet:
        lambdau: The penalty factor over all penalty terms, default=0.1.
        alpha: The ratio of L1 penalisation to L2 penalisation, default=0 (ridge).
        standardize: Indicates whether features should be standardized
            before estimation, default=False.
        intercept: Indicates whether to include an intercept, default=False.
        threshold: Optimisation convergence threshold, defaut=1e-4.
        max_iter: Maximum number of iterations, default=1e5.

    """

    def __init__(
        self,
        alpha: float = 0.1,
        lambdau: float = 0.1,
        gamma: float = 1.0,
        ini_alpha: float = 0,  # 1e-4,
        ini_lambdau: float = None,
        penalty_weights: np.ndarray = None,
        **kwargs,
    ):
        ElasticNet.__init__(
            self,
            alpha=alpha,
            lambdau=lambdau,
            **kwargs,
        )
        self.ini_alpha = ini_alpha
        self.ini_lambdau = ini_lambdau
        self.penalty_weights = penalty_weights
        self._gamma = gamma

    @property
    def gamma(self):
        """The exponent used to scale the penalty weights, default=1."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        # Penalty weights are updated if new gamma is set.
        if self.penalty_weights is not None and hasattr(self, "_gamma"):
            self.penalty_weights = self.penalty_weights ** (gamma / self.gamma)
        # set gamma
        self._gamma = gamma

    def _guess_grid(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logs: bool = True,
        n_values: int = 20,
    ) -> dict:
        """Guesses a plausible grid for the penalty hyperparameter lambda.

        The choice of grid depends on init_alpha and the standard deviation of
        the labels y. The grid can be linear, geometric, or a combination of both.

        Args:
            X: The input samples of shape (n_samples, k_features),
                can be a sparse matrix.
            y: Labels of shape (n_samples,) corresponding to the inputs X.
            logs: Indicates if the grid is geometric, default=True. Both grids
                will be considered if set to None.
            n_values: The number of points in the grid.

        Returns:
            grid: Dictionary with candidate hyperparameter values for alpha
                and lambdau.

        """
        # limits
        lower = y.std() / X.shape[1]
        upper = y.std() * X.shape[1]

        # consider linear scale or geometric or both
        if logs is None:
            lambdau_grid = np.unique(
                np.concatenate(
                    [
                        np.geomspace(lower, upper, n_values),
                        np.linspace(lower, upper, n_values),
                    ]
                )
            )
        elif logs:
            lambdau_grid = np.geomspace(lower, upper, n_values)
        else:
            lambdau_grid = np.linspace(lower, upper, n_values)
        grid = {"alpha": [self.ini_alpha], "lambdau": lambdau_grid}
        return grid

    def _update_penalty_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty_weights: np.ndarray = None,
        grid: dict = None,
        split: int = 5,
        **kwargs,
    ) -> None:
        """Updates the penalty_weights attribute using a first-stage elastic net.

        If lambda is not set explicitly, cross-validation is performed to find it.

        Args:
            X: The input samples of shape (n_samples, k_features),
                can be a sparse matrix.
            y: Labels of shape (n_samples,) corresponding to the inputs X.
            penalty_weights: Coefficient penalty weights, zero if not penalised,
                of shape (k_features,), default=None.
            grid: Dictionary with candidate hyperparameter values for alpha
                and lambdau.
            split: Number of splits to use for cross-validation.

        """
        if penalty_weights is not None:
            penalise = penalty_weights != 0
        else:
            penalise = None

        # initialise first-stage net
        ini_net = ElasticNet(alpha=self.ini_alpha, lambdau=self.ini_lambdau)

        if self.ini_lambdau is not None:
            # fit initialising net for given hyperparmeters
            ini_coef = ini_net.fit(X, y, penalty_weights=penalise, **kwargs).coef_

        else:
            # perform cross-validation on initialising net hyperparmeters
            print("Searching suitable init_lambda hyperparameter...")
            if grid is None:
                grid = self._guess_grid(X, y, logs=None, n_values=25)
            cv = GridSearchCV(ini_net, grid, cv=split, n_jobs=-1)
            cv.fit(X, y, penalty_weights=penalise, **kwargs)
            ini_coef = cv.best_estimator_.coef_
            self.ini_lambdau = cv.best_params_["lambdau"]

        # create penalty weights
        penalty_weights = abs(ini_coef.ravel() + 1 / len(y)) ** -self.gamma
        if penalise is not None:
            penalty_weights *= penalise

        self.penalty_weights = penalty_weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty_weights: np.ndarray = None,
        force_update: bool = False,
        return_fit: bool = False,
        ini_grid: dict = None,
        ini_split: int = 5,
        **kwargs,
    ):
        """Fits the model parameters to input data.

        Will perform a two-step estimation, where the first step is
        cross-validated if penalty weights are not explicitly fixed.

        Args:
            X: The input samples of shape (n_samples, k_features),
                can be a sparse matrix.
            y: Labels of shape (n_samples,) corresponding to the inputs X.
            penalty_weights: Coefficient penalty weights, zero if not penalised,
                of shape (k_features,), default=None. Note that passing
                this parameter only defines which ceofficients are not penalised.
                To fix the penalty weights levels, use object attribute instead.
            force_update: If set True, forces penalty weights to be updated,
                default=False.
            return_fit: Indicates whether full fit statistics are returned
                instead of fitting model inplace, default=False.
            ini_grid: Dictionary with candidate hyperparameter values for alpha
                and lambdau in the initialising estimation.
            split: Defines cross-validation sample splitting for the initilising
                estimation.
                Can be passed sklearn.model_selection._split.BaseCrossValidator.

        Returns:
            self : The fitted AdaptiveElasticNet object.
            fit (dict): The glmnet results as obtained from the glmnet routine
                of the second estiamtion step.

        """
        # update penalty weights if not available or user forced
        if self.penalty_weights is None or force_update:
            print("Updating penalty_weights...")
            self._update_penalty_weights(
                X, y, penalty_weights=penalty_weights, grid=ini_grid, split=ini_split
            )

        # fit using parent class method with pre-defined penalty weights
        fit = ElasticNet.fit(
            self,
            X,
            y,
            penalty_weights=self.penalty_weights,
            return_fit=return_fit,
            **kwargs,
        )

        # returns
        if return_fit:
            return fit
        else:
            return self
