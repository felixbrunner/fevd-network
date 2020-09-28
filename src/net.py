import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error

import glmnet_python
from glmnet import glmnet


class ElasticNet(BaseEstimator):
    ''' Elastic Net estimator based on the Fortran routing glmnet.
    
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
    >>> from srcnet import ElasticNet
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
            Indicates whether full fit statistics are returned.
        
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
        #X, y = check_X_y(X, y, accept_sparse=True)
        
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
        
        # results
        self.coef_ = fit['beta']
        self.R2_ = fit['dev']
        self.df_used_ = fit['df']
        self.is_fitted_ = True
        if return_fit:
            return self, fit
        else:
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
    
    def score(self, X, y):
        '''Returns negative of MSE for input data.'''
        predictions = self.predict(X)
        nmse = -mean_squared_error(predictions, y)
        return nmse
    
    
def get_lambda_path(X, y, penalty_weights, nlambda=10, **kwargs):
    '''Returns the lambda path chosen by the glmnet routine.'''
    fit = glmnet(x=X, y=y, nlambda=nlambda, penalty_factor=penalty_weights, **kwargs)
    return fit['lambdau']