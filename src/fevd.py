import numpy as np
from numpy.lib.function_base import vectorize
import scipy as sp
import networkx as nx


class FEVD:
    """Forecast Error Variance Decomposition."""

    def __init__(self, var_matrices, error_cov):
        self.var_matrices = var_matrices
        self.error_cov = error_cov

    @property
    def var_matrices(self):
        """A list of np.arrays."""
        return self._var_matrices

    @var_matrices.setter
    def var_matrices(self, var_matrices):
        if type(var_matrices) != list:
            var_matrices = [var_matrices]
        self._check_var_matrices(var_matrices)
        self._var_matrices = var_matrices

    def _check_var_matrices(self, var_matrices):
        """Checks type and dims of VAR matrices"""
        for var_matrix in var_matrices:
            assert type(var_matrix) == np.ndarray, "VAR matrices must be numpy arrays"
            assert (
                var_matrix.shape[0] == var_matrix.shape[1]
            ), "VAR matrices must be square"

    @property
    def error_cov(self):
        """A list of np.arrays."""
        return self._error_cov

    @error_cov.setter
    def error_cov(self, error_cov):
        error_cov = np.array(error_cov)
        assert (
            error_cov.shape[0] == error_cov.shape[1]
        ), "Error covariance matrix must be square"
        self._error_cov = error_cov

    @property
    def n_series(self):
        """The number of series."""
        n_series = self.error_cov.shape[0]
        return n_series

    @property
    def p_lags(self):
        """The order of the VAR(p)."""
        p_lags = len(self.var_matrices)
        return p_lags

    def vma_matrix(self, horizon):
        """Returns VMA coefficient matrix corresponding to
        the set of input VAR coefficient matrices and an input horizon.
        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

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
                phi_h += self.var_matrices[l] @ self.vma_matrix(horizon - l - 1)
            return phi_h

    def impulse_response(self, horizon):
        """Returns the h-step impulse response function
        to a generalised impulse.
        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # diagonal impulses
        diag_sigma = np.diag(np.diag(self.error_cov) ** -0.5)

        # transmission matrix
        psi_h = self.vma_matrix(horizon) @ self.error_cov @ diag_sigma
        return psi_h

    def innovation_response_variance(self, horizon):
        """Returns the sum of the h-period covariance of observations to h
        innovations - the innovation response variance (IRV).
        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # accumulate period-wise covariance contributions
        irv = np.zeros(self.error_cov.shape)
        for h in range(horizon):
            irv += self.vma_matrix(h) @ self.error_cov

        return irv

    def fev_single(self, horizon):
        """Returns the h-step ahead forecast error variance matrix
        to generalized impulses to each variable in isolation
        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # initialise
        n_series = self.n_series
        fev_single = np.zeros([n_series, n_series])

        # accumulate
        for h in range(horizon + 1):
            fev_single += self.impulse_response(h) ** 2
        return fev_single

    def fev_total(self, horizon):
        """Returns the h-step ahead forecast MSE to
        a generalised impulse to all variables.
        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # initialise
        n_series = self.n_series
        fev_total = np.zeros([n_series, n_series])

        # accumulate
        for h in range(horizon + 1):
            phi_h = self.vma_matrix(h)
            fev_total += phi_h @ self.error_cov @ phi_h.T

        fev_total = np.diag(fev_total).reshape(-1, 1)
        return fev_total

    def fev_others(self, horizon):
        """Returns the h-step ahead forecast error variance
        total contributions from other variables.
        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # initialise
        fev_single = self.fev_single(horizon=horizon)

        # sum & deduct own contribution
        fev_others = (fev_single.sum(axis=1) - np.diag(fev_single)).reshape(-1, 1)
        return fev_others

    def fev_self(self, horizon):
        """Returns the h-step ahead forecast error variance
        total contributions from lags of own timeseries.
        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # initialise
        fev_single = self.fev_single(horizon=horizon)

        # sum & deduct own contribution
        fev_self = np.diag(fev_single).reshape(-1, 1)
        return fev_self

    def fu_single(self, horizon):
        """Returns the h-step ahead forecast uncertainty matrix
        to generalized impulses to each variable in isolation
        """
        fu_single = self.fev_single(horizon) ** 0.5
        return fu_single

    def fu_total(self, horizon=1):
        """Returns the h-step ahead forecast uncertainty to
        a generalised impulse to all variables.
        """
        fu_total = self.fev_total(horizon) ** 0.5
        return fu_total

    def fu_others(self, horizon):
        """Returns the h-step ahead forecast uncertainty
        total contributions from other variables.
        """
        # initialise
        fu_single = self.fu_single(horizon=horizon)

        # sum & deduct own contribution
        fu_others = (fu_single.sum(axis=1) - np.diag(fu_single)).reshape(-1, 1)
        return fu_others

    def fu_self(self, horizon):
        """Returns the h-step ahead forecast error variance
        total contributions from lags of own timeseries.
        """
        # initialise
        fu_single = self.fu_single(horizon=horizon)

        # sum & deduct own contribution
        fu_self = np.diag(fu_single).reshape(-1, 1)
        return fu_self

    def decompose_fev(self, horizon, normalise=False):
        """Returns the forecast MSE decomposition matrix at input horizon."""
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        decomposition = self.fev_single(horizon) / self.fev_total(horizon)

        # row normalise if requested
        if normalise:
            decomposition /= decomposition.sum(axis=1).reshape(-1, 1)
        return decomposition

    def decompose_fu(self, horizon, normalise=False):
        """Returns the forecast MSE decomposition matrix at input horizon."""
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        decomposition = self.fu_single(horizon) / self.fu_total(horizon)

        # row normalise if requested
        if normalise:
            decomposition /= decomposition.sum(axis=1).reshape(-1, 1)
        return decomposition

    def in_connectedness(self, horizon, normalise=False, network="fev"):
        """"""
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"
        assert network in ["fev", "fu"], "network needs to be either fev or fu"

        if network == "fev":
            decomposition_pct = self.decompose_fev(horizon, normalise=normalise)
        elif network == "fu":
            decomposition_pct = self.decompose_fu(horizon, normalise=normalise)

        in_connectedness = decomposition_pct.sum(axis=1) - np.diag(decomposition_pct)
        in_connectedness = in_connectedness.reshape(-1, 1)
        return in_connectedness

    def out_connectedness(self, horizon, normalise=False, network="fev"):
        """"""
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"
        assert network in ["fev", "fu"], "network needs to be either fev or fu"

        if network == "fev":
            decomposition_pct = self.decompose_fev(horizon, normalise=normalise)
        elif network == "fu":
            decomposition_pct = self.decompose_fu(horizon, normalise=normalise)

        out_connectedness = decomposition_pct.sum(axis=0) - np.diag(decomposition_pct)
        out_connectedness = out_connectedness.reshape(-1, 1)
        return out_connectedness

    def average_connectedness(self, horizon, normalise=False, network="fev"):
        """"""
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"
        assert network in ["fev", "fu"], "network needs to be either fev or fu"

        in_connectedness = self.in_connectedness(
            horizon, normalise=normalise, network=network
        )
        average_connectedness = in_connectedness.mean()
        return average_connectedness

    def in_entropy(self, horizon, normalise=True, network="fev"):
        """Returns the row-wise entropy of connections.s"""
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"
        assert network in ["fev", "fu"], "network needs to be either fev or fu"

        if network == "fev":
            decomposition_pct = self.decompose_fev(horizon, normalise=normalise)
        elif network == "fu":
            decomposition_pct = self.decompose_fu(horizon, normalise=normalise)

        # remove diagonal values & scale
        n = decomposition_pct.shape[0]
        decomposition_pct = decomposition_pct[~np.eye(n, dtype=bool)].reshape(n, n - 1)
        decomposition_pct /= decomposition_pct.sum(axis=1).reshape(n, 1)

        in_entropy = sp.stats.entropy(decomposition_pct, axis=1, base=n - 1)
        in_entropy = in_entropy.reshape(-1, 1)
        return in_entropy

    def summarize(self, horizon):
        """Returns a summarising dictionary."""
        summary_dict = {
            "average_connectedness": self.average_connectedness(horizon=horizon),
            "in_connectedness": self.in_connectedness(horizon=horizon),
            "out_connectedness": self.out_connectedness(horizon=horizon),
            "fev_others": self.fev_others(horizon=horizon),
            "fev_self": self.fev_self(horizon=horizon),
            "fev_total": self.fev_total(horizon=horizon),
            "fev_single_sum": self.fev_single(horizon=horizon)
            .sum(axis=1)
            .reshape(-1, 1),
        }
        return summary_dict

    def to_fev_graph(self, horizon=1, normalise=True):
        """Returns a networkx Graph object from
        FEV decomposition at input horizon.
        """
        adjacency = self.decompose_fev(horizon=horizon, normalise=normalise)
        graph = nx.convert_matrix.from_numpy_array(adjacency, create_using=nx.DiGraph)
        return graph

    def to_fu_graph(self, horizon=1, normalise=True):
        """Returns a networkx Graph object from
        FU decomposition at input horizon."""
        adjacency = self.decompose_fu(horizon=horizon, normalise=normalise)
        graph = nx.convert_matrix.from_numpy_array(adjacency, create_using=nx.DiGraph)
        return graph

    @property
    def generalized_error_cov(self):
        """Returns the generalized innovation covariance matrix."""
        omega = (
            np.diag(np.diag(self.error_cov)) ** 0.5
            @ np.linalg.inv(self.error_cov)
            @ np.diag(np.diag(self.error_cov)) ** 0.5
        )
        return omega

    def test_diagonal_generalized_innovations(
        self, t_observations: int, method: str = "ledoit-wolf"
    ):
        """Calculate a chi2 test statistic for the null hypothesis
        H0: Omega = Identity.
        The test method can either be 'ledoit-wolf' or 'likelihood-ratio.
        Returns a tuple (test_statistic, p_value).
        """
        assert method in [
            "ledoit-wolf",
            "likelihood-ratio",
        ], "available methods are ledoit-wolf and likelihood-ratio"

        omega = self.generalized_error_cov

        if method == "ledoit-wolf":
            test_statistic = (
                N
                * T
                / 2
                * (
                    1 / N * np.trace(np.linalg.matrix_power(omega - np.eye(N), 2))
                    - N / T * (1 / N * np.trace(omega)) ** 2
                    + N / T
                )
            )
        elif method == "likelihood-ratio":
            df = t_observations - 1
            v = df * (
                np.log(self.n_series)
                - np.log(np.linalg.eigh(omega)[0].sum())
                + np.trace(omega)
                - self.n_series
            )
            test_statistic = (
                1 - 1 / (6 * df - 1) * (2 * self.n_series + 1 - 2 / (self.n_series + 1))
            ) * v

        p_value = 1 - sp.stats.chi2.cdf(
            test_statistic, self.n_series * (self.n_series + 1) / 2
        )
        return (test_statistic, p_value)
