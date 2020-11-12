import numpy as np
import scipy as sp
import pandas as pd

from sklearn.base import BaseEstimator
from src.net import ElasticNet
from sklearn.linear_model import LinearRegression


    
def soft_threshold_rule(self, x, thresholds, **kwargs):
    '''The soft thresholding rule of Cai/Liu (2011).
    Note that x is typically a covariance matrix.
    '''
    diff = abs(x)-thresholds
    estimate = np.sign(x) * diff * (diff>0)
    return estimate

def adaptive_LASSO_threshold_rule(self, x, thresholds, eta=1, **kwargs):
    '''The LASSO thresholding rule of Cai/Liu (2011).
    Note that x is typically a covariance matrix.
    '''
    diff = (1-abs(thresholds/x)**eta)
    estimate = x * diff * (diff>0)
    return estimate

    
class AdaptiveThresholdEstimator(BaseEstimator):
    '''AdaptiveThresholdEstimator object.
    Currently employs the simple thresholding level of Cai/Liu (2011)
    and the adaptive LASSO threshold rule. If the eta parameter is equal
    to 1 the estimator is equivalent to the soft-thresholding rule.
    '''
    
    def __init__(self, delta=0.95, eta=1, **kwargs):
        self.delta = delta
        self.eta = eta
    
    @property
    def confidence_threshold(self):
        confidence_threshold = sp.stats.norm.ppf(self.delta)
        return confidence_threshold

    def _t_periods(self, data):
        '''Returns the number of periods in the input data.'''
        t_periods = data.shape[0]
        return t_periods
    
    def _n_series(self, data):
        '''Returns the number of series in the input data.'''
        n_series = data.shape[1]
        return n_series
    
    def _sample_cov(self, data):
        '''Returns the sample covariance matrix of input data.'''
        cov = np.cov(data, rowvar=False)
        return cov
    
    def _var_of_cov(self, data):
        '''Returns the variances of the entries in the sample covariance
        matrix shaped equally as the covariance matrix.
        '''
        # inputs
        cov = np.cov(data, rowvar=False)
        data -= data.mean(axis=0)
        n_series = self._n_series(data)
        
        # squared deviations from mean
        deviations = np.kron(data, np.ones(n_series)) * np.kron(np.ones(n_series), data)
        
        # squared deviations from sample covariance matix
        var = (deviations - cov.reshape(1, -1))**2
        
        # output
        cov_var = var.mean(axis=0).reshape(n_series, n_series)
        return cov_var
        
    def _mean(self, data):
        '''Returns the timeseries means of the data'''
        mean = data.mean(axis=0)
        return mean
    
    def thresholds(self, data):
        '''The simple thresholding levels of Cai/Liu (2011).'''
        # inputs
        log_n_series = np.log(self._n_series(data))
        t_periods = self._t_periods(data)
        var_of_cov = self._var_of_cov(data)
        
        # calculate thresholds
        thresholds = self.confidence_threshold * np.sqrt(var_of_cov*log_n_series/t_periods)
        return thresholds
    
    def _soft_threshold_rule(self, data):
        '''The soft thresholding rule of Cai/Liu (2011).'''
        sample_cov = self._sample_cov(data)
        diff = abs(sample_cov)-self.thresholds(data)
        estimate = np.sign(sample_cov) * diff * (diff>0)
        return estimate

    def _adaptive_LASSO_threshold_rule(self, data):
        '''The LASSO thresholding rule of Cai/Liu (2011).'''
        sample_cov = self._sample_cov(data)
        diff = (1-abs(self.thresholds(data)/sample_cov)**self.eta)
        estimate = sample_cov * diff * (diff>0)
        return estimate
    
    def fit(self, X, y=None, **kwargs):
        '''Fits the AdaptiveThresholdEstimator to the data in X.
        The y input is not considered in the calculations.
        '''
        estimates = self._adaptive_LASSO_threshold_rule(X, **kwargs)
        self.covar_ = estimates
        
    def _mean_period_loss(self, estimate, data):
        '''Frobenius norm loss wrt individual periods.'''
        # setup
        t_periods = data.shape[0]
        n_series = data.shape[1]
        ones = np.ones([1, n_series])
    
        # Xt.T times Xt => squared observations
        XX = np.kron(data, ones) * np.kron(ones, data)
    
        # individual losses
        obs_losses = (estimate.reshape(1,-1) - XX)
    
        # Frobenius norm
        period_losses = (obs_losses**2).sum(axis=1)**0.5
    
        # aggregated loss
        loss = t_periods**-1 * period_losses.sum()
        return loss
    
    def score(self, X, y=None):
        '''Scores a sample using the Frobenius norm loss on
        individual periods with the fitted covariance matrix.'''
        loss = -1*self._mean_period_loss(self.covar_, X)
        return loss
    

        
        
        
        
        
##############################################################################################
        
        

class CorrelationNet():
    '''A test of the regression correlation estimator.'''
    
    def __init__(self, alpha=0.5, lambdau=1):
        self.alpha = alpha
        self.lambdau = lambdau
        
    def _n_series(self, data):
        n_series = data.shape[1]
        return n_series
    
    def _t_periods(self, data):
        t_periods = data.shape[0]
        return t_periods
    
    def _demean(self, data):
        demeaned = data - data.mean(axis=0)
        return demeaned
        
    def _build_X(self, data):
        n_series = self._n_series(data)
        t_periods = self._t_periods(data)
        X = sp.sparse.kron(sp.sparse.eye(n_series**2), np.ones([t_periods,1]))
        return X
    
    def _build_y(self, data):
        n_series = self._n_series(data)
        ones = np.ones([1, n_series])
        demeaned  = self._demean(data)
        interactions = np.kron(demeaned, ones) * np.kron(ones, demeaned)
        signs = np.sign(interactions)
        y = signs.reshape(-1, 1, order='F')
        return y
    
    def _build_X_y(self, data):
        X = self._build_X(data)
        y = self._build_y(data)
        return (X, y)
            
    def fit_OLS(self, data, return_model=False):
        ''''''
        # build inputs
        X, y = self._build_X_y(data)
        
        # estimate
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        if return_model:
            return model
        else:
            self.coef_ =  model.coef_.ravel()
            
    def fit_elastic_net(self, data, alpha=0.5, lambdau=1, penalty_weights=None, return_model=False, **kwargs):
        '''Fits the FAVAR coefficients using the glmnet routine.'''
        # build inputs
        X, y = self._build_X_y(data)
        
        # estimate
        model = ElasticNet(alpha=alpha, lambdau=lambdau, intercept=False, standardize=False, **kwargs)
        model.fit(X, y, penalty_weights=penalty_weights)
        
        if return_model:
            return model
        else:
            self.coef_ =  model.coef_.ravel()