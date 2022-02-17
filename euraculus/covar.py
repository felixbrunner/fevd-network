import numpy as np
import scipy as sp
import pandas as pd

from sklearn.base import BaseEstimator
from euraculus.net import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.covariance import GraphicalLasso

    
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
        if confidence_threshold == np.inf:
            confidence_threshold = 1e22
        return confidence_threshold

    @staticmethod
    def _t_periods(data):
        '''Returns the number of periods in the input data.'''
        t_periods = data.shape[0]
        return t_periods
    
    @staticmethod
    def _n_series(data):
        '''Returns the number of series in the input data.'''
        n_series = data.shape[1]
        return n_series
    
    @staticmethod
    def _sample_cov(data):
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
        deviations = np.einsum('ti, tj -> ijt', data, data)
        
        # squared deviations from sample covariance matix
        var = (deviations - cov.reshape(n_series, n_series, 1))**2
        
        # output
        cov_var = var.mean(axis=2)
        return cov_var
    
    @staticmethod
    def _mean(data):
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
        self.covariance_ = estimates

    def _covar_loss(self, data):
        '''Frobenius norm loss wrt sample covariance matrix.'''
        # differences
        diff = self.covariance_ - self._sample_cov(data)
    
        # Frobenius norm
        loss = (diff**2).sum()
    
        return loss
        
    def _mean_period_loss(self, data, exclude_diag=False):
        '''Frobenius norm loss wrt individual periods.'''
        # setup
        n_series = data.shape[1]
        t_periods = data.shape[0]
        
        # Xt.T times Xt => squared observations
        XX = np.einsum('ti, tj -> ijt', data, data)
        obs_losses = self.covariance_.reshape(n_series, n_series, 1) - XX
        
        # aggregate loss
        total_loss = np.triu((obs_losses**2).transpose((2, 0, 1)), k=exclude_diag).transpose((1, 2, 0)).sum()
        mean_loss = total_loss/(t_periods*n_series*(n_series-1)/2)
        
        return mean_loss
    
    def score(self, X, y=None, scorer='mean_period'):
        '''Scores a sample using the Frobenius norm loss on
        individual periods with the fitted covariance matrix.'''
        if scorer == 'covar':
            loss = -1*self._covar_loss(X)
        elif scorer == 'mean_period':
            loss = -1*self._mean_period_loss(X)
        else:
            raise Exception('Scoring function not implemented')
        return loss

    
class GLASSO(GraphicalLasso):
    
    def _covar_loss(self, data):
        '''Frobenius norm loss wrt sample covariance matrix.'''
        # differences
        diff = self.covariance_ - self._sample_cov(data)
    
        # norm
        loss = (diff**2).sum()
    
        return loss
        
    def _mean_period_loss(self, data, exclude_diag=False):
        '''Frobenius norm loss wrt individual periods.'''
        # setup
        n_series = data.shape[1]
        t_periods = data.shape[0]
        
        # Xt.T times Xt => squared observations
        XX = np.einsum('ti, tj -> ijt', data, data)
        obs_losses = self.covariance_.reshape(n_series, n_series, 1) - XX
        
        # aggregate loss
        total_loss = np.triu((obs_losses**2).transpose((2, 0, 1)), k=exclude_diag).transpose((1, 2, 0)).sum()
        mean_loss = total_loss/(t_periods*n_series*(n_series-1)/2)
        
        return mean_loss
    
    def score(self, X, y=None, scorer='mean_period'):
        '''Scores a sample using the Frobenius norm loss on
        individual periods with the fitted covariance matrix.'''
        if scorer == 'covar':
            loss = -1*self._covar_loss(X)
        elif scorer == 'mean_period':
            loss = -1*self._mean_period_loss(X)
        else:
            raise Exception('Scoring function not implemented')
        return loss
    
    @property
    def covariance_density_(self):
        '''Returns the density of the estimated covariance matrix.'''
        density = np.count_nonzero(self.covariance_)/np.size(self.covariance_)
        return density
    
    @property
    def covariance_sparsity_(self):
        '''Returns the sparsity of the estimated covariance matrix.'''
        sparsity = 1 - self.covariance_density_
        return sparsity
        
    @property
    def precision_density_(self):
        '''Returns the density of the estimated precision matrix.'''
        density = np.count_nonzero(self.precision_)/np.size(self.precision_)
        return density
    
    @property
    def precision_sparsity_(self):
        '''Returns the sparsity of the estimated precision matrix.'''
        sparsity = 1 - self.precision_density_
        return sparsity
       
        
        
        
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