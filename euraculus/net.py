import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

import glmnet_python
from glmnet import glmnet

import copy


class ElasticNet(BaseEstimator):
    ''' Elastic Net estimator based on the Fortran routine glmnet.
    
    Parameters
    ----------
    lambdau : float, default=0.1
        The penalty factor over all penalty terms.
    alpha : float, default=0.1
        The ratio of L1 penalisation to L2 penalisation.
    standardize : bool, default=False
        Indicates whether features should be standardised
        before estimation.
    intercept : bool, default=False
        Indicates whether to include an intercept.
    threshold : float, defaut=1e-4
        Optimisation convergence threshold.
    max_iter : int, default 1e5
        Maximum number of iterations.
    
    Examples
    --------
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
    '''
    
    def __init__(self,
                 lambdau=0.1,
                 alpha=0.1,
                 standardize=False,
                 intercept=False,
                 threshold=1e-4,
                 max_iter=1e5,
                 **kwargs):
        
        self.lambdau = lambdau
        self.alpha = alpha
        self.standardize = standardize
        self.intercept = intercept
        self.threshold = threshold
        self.max_iter = max_iter
        
        self.__dict__.update(kwargs)
    
    def _fix_data(self, data):
        '''Sets appropriate types for data inputs'''
        data = data.astype('float64')
        if sp.sparse.issparse(data):
            data = data.tocsc()
        return data
        
    def fit(self, X, y, penalty_weights=None, return_fit=False, **kwargs):
        '''Fits the model parameters to input data.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        penalty_weights : array-like, shape (n_features,), default=None
            Coefficient penalty weights, zero if not penalised.
        return_fit: bool, default=False
            Indicates whether full fit statistics are returned instead of 
            fitting model inplace.
        
        Returns
        -------
        self : object
            Returns self.
        fit : dict
            glmnet results.
        '''
        # dimensions
        assert X.shape[0] == y.shape[0], \
            'data dimension mismatch'
        
        n_obs = y.shape[0]
        n_feat = X.shape[1]
        
        # set penalty weights
        if penalty_weights is None:
            penalty_weights = np.ones([n_feat])
            
        # transform inputs for glmnet
        lambdau = np.array([self.lambdau])
        X = self._fix_data(X)
        y = self._fix_data(y)
        
        # estimate
        fit = glmnet(x=X,
                     y=y,
                     alpha=self.alpha,               # corresponds to kappa hyperparameter
                     standardize=self.standardize,   # standardise data before optimisation
                     lambdau=lambdau,                # lambda hyperparameter
                     penalty_factor=penalty_weights, # coefficient penalty weight
                     intr=self.intercept,            # intercept
                     thresh=self.threshold,          # convergence threshold
                     maxit=self.max_iter,            # maximum number of iterations
                    )
        
        if return_fit:
            return fit
        else:
            # store results
            self.coef_ = fit['beta']
            self.R2_ = fit['dev'][0]/100
            self.df_used_ = fit['df'][0]
            self.is_fitted_ = True
            return self
        
    def predict(self, X):
        '''Predicts y given X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        '''
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        y = X @ self.coef_
        return y
    
    def score(self, X, y, scorer=None):
        '''Returns negative of MSE for input data.'''
        predictions = self.predict(X)
        if scorer is None:
            score = -mean_squared_error(predictions, y)
        else:
            score = -scorer(predictions, y)
        return score
    
    def copy(self):
        '''Returns a copy of self'''
        return copy.deepcopy(self)
    
    
#     def find_lambda_path(self, X, y, alpha=None, penalty_weights=None, nlambda=10, **kwargs):
#         '''Uses glmnet API to find a plausible path for lambda.
#         ''' 
#         # dimensions
#         assert X.shape[0] == y.shape[0], \
#             'data dimension mismatch'
        
#         n_obs = y.shape[0]
#         n_feat = X.shape[1]
        
#         # set penalty weights
#         if penalty_weights is None:
#             penalty_weights = np.ones([n_feat])
            
#         # set alpha
#         if alpha is None:
#             alpha = self.alpha        
     
#         # transform inputs for glmnet
#         X = self._fix_data(X)
#         y = self._fix_data(y)
        
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
        
#         # set grid
#         if type(lambda_grid) == np.ndarray:
#             lambda_grid = lambda_grid.tolist()
#         grid = {'lambdau': lambda_grid}
        
#         # set split
#         if split is None:
#             split = PredefinedSplit([0, 1]*(len(y)//2))
#         cv = GridSearchCV(elnet, grid, cv=split, **kwargs)
        
#         # fit
#         cv.fit(X, y, penalty_weights=penalty_weights, refit=False)
        
#         # extract
#         lambdau = grid['lambdau'][cv.best_index_]
            
#         return lambdau
    
    
class AdaptiveElasticNet(ElasticNet):
    '''Implementation of the Adaptive Elastic Net Estimator of Zou/Zhang 2009.
    '''
    
    def __init__(self,
                 alpha=0.1,
                 lambdau=0.1,
                 gamma=1,
                 ini_alpha=0.01,
                 ini_lambdau=None,
                 penalty_weights=None,
                 **kwargs):
        ElasticNet.__init__(self, alpha=alpha, lambdau=lambdau, **kwargs)
        self.ini_alpha = ini_alpha
        self.ini_lambdau = ini_lambdau
        self.penalty_weights = penalty_weights
        self._gamma = gamma
        
    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, gamma):
        '''Penalty weights are updated if new gamma is set.'''
        if self.penalty_weights is not None and hasattr(self, '_gamma'):
            self.penalty_weights = self.penalty_weights**(gamma/self.gamma)
        self._gamma = gamma
        
    def _guess_grid(self, X, y, logs=True, n_values=20):
        '''Guesses a plausible grid for the penalty hyperparameter lambda.'''
        # limits
        lower = y.std()/X.shape[1]
        upper = y.std()*X.shape[1]
        
        # consider linear scale or geometric or both
        if logs is None:
            lambdau_grid = np.unique(np.concatenate([np.geomspace(lower, upper, n_values),\
                                                     np.linspace(lower, upper, n_values)]))
        elif logs:
            lambdau_grid = np.geomspace(lower, upper, n_values)
        else:
            lambdau_grid = np.linspace(lower, upper, n_values)
        grid = {'alpha': [self.ini_alpha],
                'lambdau': lambdau_grid}
        return grid
        
    def _update_penalty_weights(self, X, y, 
                                penalty_weights=None,
                                grid=None,
                                split=5,
                                **kwargs):
        '''Updates the penalty_weights attribute using a first-stage elastic net.
        If lambda is not set, cross-validation is performed to find it.
        '''
        if penalty_weights is not None:
            penalise = (penalty_weights != 0)
        else:
            penalise = None
        
        # initialise first-stage net
        ini_net = ElasticNet(alpha=self.ini_alpha, lambdau=self.ini_lambdau)
        
        if self.ini_lambdau is not None:
            # fit initialising net for given hyperparmeters
            ini_coef = ini_net.fit(X, y, penalty_weights=penalise, **kwargs).coef_
        
        else:
            # perform cross-validation on initialising net hyperparmeters
            print('Searching suitable init_lambda hyperparameter...')
            if grid is None:
                grid = self._guess_grid(X, y, logs=None, n_values=25)
            cv = GridSearchCV(ini_net, grid, cv=split, n_jobs=-1)
            cv.fit(X, y, penalty_weights=penalise, **kwargs)
            ini_coef = cv.best_estimator_.coef_
            self.ini_lambdau = cv.best_params_['lambdau']
            
        # create penalty weights
        penalty_weights = abs(ini_coef.ravel() + 1/len(y))**-self.gamma
        if penalise is not None:
            penalty_weights *= penalise
            
        self.penalty_weights = penalty_weights
                
    def fit(self, X, y,
            penalty_weights=None,
            force_update=False,
            return_fit=False,
            ini_grid=None,
            ini_split=5,
            **kwargs):
        '''Fits the model parameters to input data.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        penalty_weights : array-like, shape (n_features,), default=None
            Coefficient penalty weights, zero if not penalised. Note that passing
            this parameter only defines which ceofficients are not penalised. To fix
            the penalty weights levels, use object attribute instead.
        force_update: bool, default=False
            If set True, forces penalty weights to be updated.
        return_fit: bool, default=False
            Indicates whether full fit statistics are returned instead of 
            fitting model inplace.
        ini_grid: dict
            Defines the hyperparmeter grid for cross-validating the
            hyperparmeters of the initialisating estimation.
        split: sklearn.model_selection._split.BaseCrossValidator
            Defines cross-validation sample splitting.
        
        Returns
        -------
        self : object
            Returns self.
        fit : dict
            glmnet results.
        '''
        # update penalty weights if not available or user forced
        if self.penalty_weights is None or force_update:
            print('Updating penalty_weights...')
            self._update_penalty_weights(X, y,
                                         penalty_weights=penalty_weights,
                                         grid=ini_grid,
                                         split=ini_split)
        
        # fit using parent class method with pre-defined penalty weights
        fit = ElasticNet.fit(self, X ,y,
                                 penalty_weights=self.penalty_weights,
                                 return_fit=return_fit, **kwargs)
        
        # returns
        if return_fit:
            return fit
        else:
            return self