"""."""

import warnings
import networkx as nx
import numpy as np
import scipy as sp


class FEVD:
    """Forecast Error Variance Decomposition.

    Attributes:
        var_matrices: The vector auto-regression coefficient matrices.
        error_cov: The innovation covariance matrix.
        n_series: The number of series.
        p_lags: The order of the VAR(p).
        generalized_error_cov: The generalized innovation covariance matrix.

    """

    def __init__(
        self,
        var_matrices: list,
        error_cov: np.ndarray,
    ):
        """Initiates the FEVD object with attribute matrices.

        Args:
            var_matrices: The vector auto-regression coefficient matrices.
            error_cov: The innovation covariance matrix.

        """
        self.var_matrices = var_matrices
        self.error_cov = error_cov

    @property
    def var_matrices(self):
        """The VAR coefficients as a list of numpy arrays."""
        return self._var_matrices

    @var_matrices.setter
    def var_matrices(self, var_matrices: list):
        if type(var_matrices) != list:
            var_matrices = [var_matrices]
        self._check_var_matrices(var_matrices)
        self._var_matrices = var_matrices

    def _check_var_matrices(self, var_matrices: list):
        """Checks type and dims of VAR matrices.

        Args:
            var_matrices: The vector auto-regression coefficient matrices.

        """
        for var_matrix in var_matrices:
            assert type(var_matrix) == np.ndarray, "VAR matrices must be numpy arrays"
            assert (
                var_matrix.shape[0] == var_matrix.shape[1]
            ), "VAR matrices must be square"

    @property
    def error_cov(self):
        """The innovation covariance matrix as a numpy array."""
        return self._error_cov

    @error_cov.setter
    def error_cov(self, error_cov: np.ndarray):
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

    def vma_matrix(self, horizon: int) -> np.ndarray:
        """Invert the VAR to obtain MA coefficients.

        Returns a VMA coefficient matrix corresponding to the VAR
        coefficients and an input horizon.

        Args:
            horizon: Number of periods for moving average coefficients.

        Returns:
            phi_h: h-step VMA matrix (n_series * n_series).

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

    def impulse_response_functions(self, horizon: int) -> np.ndarray:
        """Calculate h-step impulse response function matrix.

        Returns the h-step impulse response functions af all series to
        a generalised impulse.

        Args:
            horizon: Number of periods for impulse response functions.

        Returns:
            psi_h: h-step impulse response matrix (n_series * n_series).

        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # diagonal impulses
        diag_sigma = np.diag(np.diag(self.error_cov) ** -0.5)

        # transmission matrix
        psi_h = self.vma_matrix(horizon) @ self.error_cov @ diag_sigma
        return psi_h

    def innovation_response_variances(self, horizon: int) -> np.ndarray:
        """Calculate h-step innovation response variance matrix.

        Returns the sum of the h-period covariance of observations to h
        innovation vectors - the innovation response variances (IRV).
        IRV = Sum_{h=0}^{H-1} VMA_h Sigma

        Args:
            horizon: Number of periods for innovation response variances.

        Returns:
            irv_h: h-step innovation response variances (n_series * n_series).

        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # accumulate period-wise covariance contributions
        irv_h = np.zeros(self.error_cov.shape)
        for h in range(horizon):
            irv_h += self.vma_matrix(h) @ self.error_cov

        return irv_h

    def forecast_error_variances(self, horizon: int) -> np.ndarray:
        """Calculate h-step forecast error variance matrix.

        Returns the h-step ahead forecast error variance matrix
        to generalized impulses to each variable in isolation.

        Args:
            horizon: Number of periods for forecast error variances.

        Returns:
            fev_h: h-step forecast error variance matrix (n_series * n_series).

        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # initialise
        n_series = self.n_series
        fev_h = np.zeros([n_series, n_series])

        # accumulate
        for h in range(horizon + 1):
            fev_h += self.impulse_response_functions(h) ** 2
        return fev_h

    def mean_squared_errors(self, horizon: int) -> np.ndarray:
        """Calculate h-step mean squared error of each series.

        Returns the h-step ahead forecast MSE to
        a generalised impulse to all variables.

        Args:
            horizon: Number of periods for the mean squared errors.

        Returns:
            mse_h: h-step mean squared error vector (n_series * 1).

        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        # initialise
        n_series = self.n_series
        mse_h = np.zeros([n_series, n_series])

        # accumulate
        for h in range(horizon + 1):
            phi_h = self.vma_matrix(h)
            mse_h += phi_h @ self.error_cov @ phi_h.T

        mse_h = np.diag(mse_h).reshape(-1, 1)
        return mse_h

    def forecast_error_variance_decomposition(
        self,
        horizon: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """Calculate the forecast error variance decomposition matrix.

        Returns the forecast MSE decomposition matrix at input horizon.

        Args:
            horizon: Number of periods for fevd.
            normalize: Indicates if matrix should be row-normalized.

        Returns:
            fevd: Forecast error variance decomposition (n_series * n_series).

        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        fevd = self.forecast_error_variances(horizon) / self.mean_squared_errors(
            horizon
        )

        # row normalize if requested
        if normalize:
            fevd /= fevd.sum(axis=1).reshape(-1, 1)
        return fevd

    def forecast_uncertainty(self, horizon: int) -> np.ndarray:
        """Calculate h-step forecast uncertainty matrix.

        Returns the h-step ahead forecast uncertainty matrix
        to generalized impulses to each variable in isolation

        Args:
            horizon: Number of periods for forecast uncertainty.

        Returns:
            fu_h: h-step forecast uncertainty matrix (n_series * n_series).

        """
        fu_h = self.forecast_error_variances(horizon) ** 0.5
        return fu_h

    def mean_absolute_error(self, horizon: int) -> np.ndarray:
        """Calculate h-step mean absolute forecast error of each series.

        Returns the h-step ahead mean absolute forecast error to
        a generalised impulse to all variables.

        Args:
            horizon: Number of periods for the mean absolute forecast error.

        Returns:
            mae_h: h-step mean absolute forecast error vector (n_series * 1).

        """
        mae_h = self.mean_squared_errors(horizon) ** 0.5
        return mae_h

    def forecast_uncertainty_decomposition(
        self,
        horizon: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """Calculate the forecast uncertainty decomposition matrix.

        Returns the forecast MAE decomposition matrix at input horizon.

        Args:
            horizon: Number of periods for fud.
            normalize: Indicates if matrix should be row-normalized.

        Returns:
            fud: Forecast uncertainty decomposition (n_series * n_series).

        """
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        fud = self.forecast_uncertainty(horizon) / self.mean_absolute_error(horizon)

        # row normalize if requested
        if normalize:
            fud /= fud.sum(axis=1).reshape(-1, 1)
        return fud

    def _get_table(
        self,
        name: str,
        horizon: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """Retrieve a connectedness table from FEVD object.

        Args:
            name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            normalize: Indicates if table should be row-normalized.

        Returns:
            table: The requested (n_series * n_series) connectedness table.

        """
        # verify inputs
        assert (
            type(horizon) == int and horizon >= 0
        ), "horizon needs to be a positive integer"

        options = ["fevd", "fev", "fud", "fu", "irv", "irf", "var", "vma"]
        assert name in options, "name needs to be one of " + ", ".join(options)

        if name not in ["fevd", "fud"] and normalize:
            warnings.warn("normalization only available for tables 'fevd' and 'fud'")

        # retrieve table
        if name == "fevd":
            table = self.forecast_error_variance_decomposition(
                horizon=horizon, normalize=normalize
            )
        if name == "fev":
            table = self.forecast_error_variances(horizon=horizon)
        if name == "fud":
            table = self.forecast_uncertainty_decomposition(
                horizon=horizon, normalize=normalize
            )
        if name == "fu":
            table = self.forecast_uncertainty(horizon=horizon)
        if name == "irv":
            table = self.innovation_response_variances(horizon=horizon)
        if name == "irf":
            table = self.impulse_response_functions(horizon=horizon)
        if name == "var":
            table = self.var_matrices[horizon - 1]
        if name == "vma":
            table = self.vma_matrix(horizon=horizon)

        return table

    def in_connectedness(
        self,
        horizon: int,
        table_name: str = "fevd",
        others_only: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """Calculate the sum of incoming links per node (row-wise).

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            others_only: Indicates wheter to include self-linkages.
            normalize: Indicates if table should be row-normalized.

        Returns:
            in_connectedness: A (n_series * 1) vector with connectedness values.

        """
        table = self._get_table(
            name=table_name,
            horizon=horizon,
            normalize=normalize,
        )

        in_connectedness = table.sum(axis=1)
        if others_only:
            in_connectedness -= np.diag(table)
        in_connectedness = in_connectedness.reshape(-1, 1)
        return in_connectedness

    def out_connectedness(
        self,
        horizon: int,
        table_name: str = "fevd",
        others_only: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """Calculate the sum of outgoing links per node (column-wise).

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            others_only: Indicates wheter to include self-linkages.
            normalize: Indicates if table should be row-normalized.

        Returns:
            out_connectedness: A (n_series * 1) vector with connectedness values.

        """
        table = self._get_table(
            name=table_name,
            horizon=horizon,
            normalize=normalize,
        )

        out_connectedness = table.sum(axis=0)
        if others_only:
            out_connectedness -= np.diag(table)
        out_connectedness = out_connectedness.reshape(-1, 1)
        return out_connectedness

    def self_connectedness(
        self,
        horizon: int,
        table_name: str = "fevd",
        normalize: bool = False,
    ) -> np.ndarray:
        """Get the links of each node with itself.

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            others_only: Indicates wheter to include self-linkages.
            normalize: Indicates if table should be row-normalized.

        Returns:
            self_connectedness: A (n_series * 1) vector with connectedness values.

        """
        table = self._get_table(
            name=table_name,
            horizon=horizon,
            normalize=normalize,
        )

        self_connectedness = np.diag(table).reshape(-1, 1)
        return self_connectedness

    def total_connectedness(
        self,
        horizon: int,
        table_name: str = "fevd",
        others_only: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """Get the total links of each node (incoming and outgoing).

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            others_only: Indicates wheter to include self-linkages.
            normalize: Indicates if table should be row-normalized.

        Returns:
            total_connectedness: A (n_series * 1) vector with connectedness values.

        """
        # collect arguments to pass
        kwargs = locals()
        kwargs.pop("self")

        # calculate
        total_connectedness = self.in_connectedness(**kwargs) + self.out_connectedness(
            **kwargs
        )
        if not others_only:
            kwargs.pop("others_only")
            total_connectedness -= self.self_connectedness(**kwargs)
        return total_connectedness

    def average_connectedness(
        self,
        horizon: int,
        table_name: str = "fevd",
        others_only: bool = True,
        normalize: bool = False,
    ) -> float:
        """Get the total links of each node (incoming and outgoing).

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            others_only: Indicates wheter to include self-linkages.
            normalize: Indicates if table should be row-normalized.

        Returns:
            average_connectedness: Average connectedness value in the table.

        """
        # collect arguments to pass
        kwargs = locals()
        kwargs.pop("self")

        # calculate
        in_connectedness = self.in_connectedness(**kwargs)
        average_connectedness = in_connectedness.mean()
        return average_connectedness

    def in_entropy(
        self,
        horizon: int,
        table_name: str = "fevd",
        others_only: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """Calculate the entropy of incoming links per node (row-wise).

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            others_only: Indicates wheter to include self-linkages.
            normalize: Indicates if table should be row-normalized.

        Returns:
            in_entropy: A (n_series * 1) vector with entropy values.

        """
        table = self._get_table(
            name=table_name,
            horizon=horizon,
            normalize=normalize,
        )
        n = self.n_series

        # remove diagonal values
        if others_only:
            table = table[~np.eye(n, dtype=bool)].reshape(n, n - 1)

        # scale rows to one
        table /= table.sum(axis=1).reshape(n, 1)

        # calculate entropy
        in_entropy = sp.stats.entropy(table, axis=1, base=n - others_only)
        in_entropy = in_entropy.reshape(-1, 1)
        return in_entropy

    def out_entropy(
        self,
        horizon: int,
        table_name: str = "fevd",
        others_only: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """Calculate the entropy of outgoing links per node (column-wise).

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            others_only: Indicates wheter to include self-linkages.
            normalize: Indicates if table should be row-normalized.

        Returns:
            out_entropy: A (n_series * 1) vector with entropy values.

        """
        table = self._get_table(
            name=table_name,
            horizon=horizon,
            normalize=normalize,
        )
        n = self.n_series

        # remove diagonal values
        if others_only:
            table = table[~np.eye(n, dtype=bool)].reshape(n - 1, n)

        # scale columns to one
        table /= table.sum(axis=0).reshape(1, n)

        # calculate entropy
        out_entropy = sp.stats.entropy(table, axis=0, base=n - others_only)
        out_entropy = out_entropy.reshape(-1, 1)
        return out_entropy

    def to_graph(
        self,
        horizon: int,
        table_name: str = "fevd",
        normalize: bool = False,
    ) -> nx.classes.digraph.DiGraph:
        """Create a networkx Graph object from a connectedness table.

        Args:
            table_name: Abbreviated name of the table.
            horizon: Number of periods to compute the table.
            normalize: Indicates if table should be row-normalized.

        Returns:
            graph: The connectedness table as a networkx DiGraph object.

        """
        table = self._get_table(
            name=table_name,
            horizon=horizon,
            normalize=normalize,
        )
        graph = nx.convert_matrix.from_numpy_array(table, create_using=nx.DiGraph)
        return graph

    @property
    def generalized_error_cov(self) -> np.ndarray:
        """The generalized innovation covariance matrix.

        Omega = diag(Sigma)^(1/2) * Sigma^(-1) * diag(Sigma)^(1/2)

        Returns:
            omega: The generalized error covariance.

        """
        omega = (
            np.diag(np.diag(self.error_cov)) ** 0.5
            @ np.linalg.inv(self.error_cov)
            @ np.diag(np.diag(self.error_cov)) ** 0.5
        )
        return omega

    def test_diagonal_generalized_innovations(
        self,
        t_observations: int,
        method: str = "ledoit-wolf",
    ) -> tuple:
        """Test diagonality of innovations.

        Calculate a chi2 test statistic for the null hypothesis
        H0: Omega = Identity.
        The test method can either be 'ledoit-wolf' or 'likelihood-ratio'.

        Args:
            t_observations (int): Number of time observations in the sample.
            method (str): The test statistic used,
                either 'ledoit-wolf' or 'likelihood-ratio'.

        Returns:
            test_statistic (float): The calculated test statistic.
            p_value (float): The corresponding p-value of the test.

        """
        assert method in [
            "ledoit-wolf",
            "likelihood-ratio",
        ], "available methods are ledoit-wolf and likelihood-ratio"

        omega = self.generalized_error_cov
        N = self.n_series
        T = t_observations

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
                np.log(N) - np.log(np.linalg.eigh(omega)[0].sum()) + np.trace(omega) - N
            )
            test_statistic = (1 - 1 / (6 * df - 1) * (2 * N + 1 - 2 / (N + 1))) * v

        p_value = 1 - sp.stats.chi2.cdf(test_statistic, N * (N + 1) / 2)
        return (test_statistic, p_value)

    def index_variance_decomposition(
        self,
        weights: np.array,
        horizon: int,
    ) -> np.array:
        """Decomposes the variance of a weighted index.

        Decomposes the variance of an index created with the input index
        weights correspoding to the FEVD constituents.
        IVD = w' * IRV * diag(w)

        Args:
            weights (numpy.array): Index weights associated with the FEVD variables.
            horizon (int): The horizon of accumulative innovations.

        Returns:
            index_variance_decomposition (np.array): index variance weights
                (n_series times 1)
        """
        assert weights.shape == (self.n_series, 1), "weights have wrong shape"

        innovation_response_variance = self.innovation_response_variances(
            horizon=horizon
        )
        index_variance_decomposition = (
            weights.T @ innovation_response_variance @ np.diag(weights)
        )
        return index_variance_decomposition.T
