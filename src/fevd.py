import numpy as np
import scipy as sp


class FEVD:
    '''Forecast Error Variance Decomposition.'''
    
    def __init__(self, var_matrices, error_cov):
        self.var_matrices = var_matrices
        self.error_cov = error_cov
        
    @property
    def var_matrices(self):
        '''A list of np.arrays.'''
        return self._var_matrices
    
    @var_matrices.setter
    def var_matrices(self, var_matrices):
        if type(var_matrices) != list:
            var_matrices = [var_matrices]
        self._check_var_matrices(var_matrices)
        self._var_matrices = var_matrices
        
    def _check_var_matrices(self, var_matrices):
        '''Checks type and dims of VAR matrices'''
        for var_matrix in var_matrices:
            assert type(var_matrix) == np.ndarray, \
                'VAR matrices must be numpy arrays'
            assert var_matrix.shape[0] == var_matrix.shape[1], \
                'VAR matrices must be square'

            
    @property
    def error_cov(self):
        '''A list of np.arrays.'''
        return self._error_cov
    
    @error_cov.setter
    def error_cov(self, error_cov):
        error_cov = np.array(error_cov)
        assert error_cov.shape[0] == error_cov.shape[1], \
            'Error covariance matrix must be square'
        self._error_cov = error_cov
    
    
    @property
    def n_series(self):
        '''The number of series.'''
        n_series = self.error_cov.shape[0]
        return n_series
    
    @property
    def p_lags(self):
        '''The order of the VAR(p).'''
        p_lags = len(self.var_matrices)
        return p_lags
    
    
    def vma_matrix(self, horizon):
        '''Returns VMA coefficient matrix corresponding to
        the set of input VAR coefficient matrices and an input horizon.
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        if horizon == 0:
            # identity
            phi_0 = np.eye(self.n_series)
            return phi_0
        
        elif self.p_lags == 1:
            # matrix power
            phi_h = np.linalg.matrix_power(self.var_matrices[0], horizon) 
            return phi_h
        
        else:           
            # initialise
            n_series = self.n_series
            phi_h = np.zeros([n_series, n_series])
            
            # recursion
            for l in range(self.p_lags):
                phi_h += self.var_matrices[l] @ self.vma_matrix(horizon-l-1)
            return phi_h

    def impulse_response(self, horizon):
        '''Returns the h-step impulse response function
        to a generalised impulse.
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        # diagonal impulses
        diag_sigma = np.diag(np.diag(self.error_cov)**-0.5)
    
        # transmission matrix
        psi_h = self.vma_matrix(horizon) @ self.error_cov @ diag_sigma
        return psi_h
        
    def fev_single(self, horizon):
        '''Returns the h-step ahead forecast error variance matrix 
        to generalized impulses to each variable in isolation
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        # initialise
        n_series = self.n_series
        fev_single = np.zeros([n_series, n_series])
        
        # accumulate
        for h in range(horizon+1):
            fev_single += self.impulse_response(h)**2
        return fev_single
    
    def fev_all(self, horizon):
        '''Returns the h-step ahead forecast MSE to
        a generalised impulse to all variables.
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        # initialise
        n_series = self.n_series
        fev_all = np.zeros([n_series, n_series])
        
        # accumulate
        for h in range(horizon+1):
            phi_h = self.vma_matrix(h)
            fev_all += phi_h @ self.error_cov @ phi_h.T
            
        fev_all = np.diag(fev_all).reshape(-1, 1)
        return fev_all

    def decompose(self, horizon):
        '''Returns the forecast MSE decomposition matrix at input horizon.
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        decomposition = self.fev_single(horizon) / self.fev_all(horizon)
        return decomposition
    
    def decompose_pct(self, horizon):
        '''Returns the percentage forecast MSE decomposition matrix at input horizon.
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        decomposition = self.decompose(horizon)
        decomposition_pct = decomposition / decomposition.sum(axis=1).reshape(-1,1)
        return decomposition_pct
    
    def in_connectedness(self, horizon):
        ''''''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
    
        decomposition_pct = self.decompose_pct(horizon)
        in_connectedness = decomposition_pct.sum(axis=1) - np.diag(decomposition_pct)
        in_connectedness = in_connectedness.reshape(-1, 1)
        return in_connectedness
    
    def out_connectedness(self, horizon):
        ''''''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
    
        decomposition_pct = self.decompose_pct(horizon)
        out_connectedness = decomposition_pct.sum(axis=0) - np.diag(decomposition_pct)
        out_connectedness = out_connectedness.reshape(-1, 1)
        return out_connectedness
    
    def average_connectedness(self, horizon):
        ''''''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
    
        in_connectedness = self.in_connectedness(horizon)
        average_connectedness = in_connectedness.mean()
        return average_connectedness
    
    def fev_others(self, horizon):
        '''Returns the h-step ahead forecast error variance 
        total contributions from other variables.
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        # initialise
        fev_single = self.fev_single(horizon=horizon)
        
        # sum & deduct own contribution
        fev_others = (fev_single.sum(axis=1) - np.diag(fev_single)).reshape(-1, 1)
        return fev_others
    
    def fev_self(self, horizon):
        '''Returns the h-step ahead forecast error variance 
        total contributions from lags of own timeseries.
        '''
        assert type(horizon) == int and horizon >= 0, \
            'horizon needs to be a positive integer'
        
        # initialise
        fev_single = self.fev_single(horizon=horizon)
        
        # sum & deduct own contribution
        fev_self = np.diag(fev_single).reshape(-1, 1)
        return fev_self
    
    def summarize(self, horizon):
        '''Returns a summarising dictionary.'''
        summary_dict = {'average_connectedness': self.average_connectedness(horizon=horizon),
                        'in_connectedness': self.in_connectedness(horizon=horizon),
                        'out_connectedness': self.out_connectedness(horizon=horizon),
                        'fev_others': self.fev_others(horizon=horizon),
                        'fev_self': self.fev_self(horizon=horizon),
                        'fev_all': self.fev_all(horizon=horizon),
                        'fev_single_sum': self.fev_single(horizon=horizon).sum(axis=1).reshape(-1,1),
                       }
        return summary_dict
    
    def to_graph(self, horizon=1):
        '''Returns a networkx Graph object from decompose_pct(horizon).'''
        from networkx.convert_matrix import from_numpy_array
        graph = from_numpy_array(self.decompose_pct(horizon=horizon))
        return graph