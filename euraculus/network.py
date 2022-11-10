""""""

import networkx as nx
import numpy as np
import scipy as sp
from euraculus.utils import herfindahl_index, power_law_exponent


class Network:
    """A Network object describes the relation of a collection of nodes."""

    def __init__(self, adjacency_matrix: np.ndarray):
        """Set up the adjacency matrix.

        Args:
            adjacency: The square matrix containing connection strength values.

        """
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError("adjacency matrix needs to be square")

        self.adjacency_matrix = adjacency_matrix

    @property
    def n_nodes(self) -> int:
        """The number of nodes in the network."""
        return self.adjacency_matrix.shape[0]

    def in_connectedness(
        self,
        others_only: bool = True,
    ) -> np.ndarray:
        """Calculate the sum of incoming links per node (row-wise).

        Args:
            others_only: Indicates wheter to include self-linkages.

        Returns:
            in_connectedness: A (n_nodes * 1) vector with connectedness values.
        """
        in_connectedness = self.adjacency_matrix.sum(axis=1)
        if others_only:
            in_connectedness -= np.diag(self.adjacency_matrix)
        in_connectedness = in_connectedness.reshape(-1, 1)
        return in_connectedness

    def out_connectedness(
        self,
        others_only: bool = True,
    ) -> np.ndarray:
        """Calculate the sum of outgoing links per node (column-wise).

        Args:
            others_only: Indicates wheter to include self-linkages.

        Returns:
            out_connectedness: A (n_nodes * 1) vector with connectedness values.
        """
        out_connectedness = self.adjacency_matrix.sum(axis=0)
        if others_only:
            out_connectedness -= np.diag(self.adjacency_matrix)
        out_connectedness = out_connectedness.reshape(-1, 1)
        return out_connectedness

    def self_connectedness(
        self,
    ) -> np.ndarray:
        """Get the links of each node with itself.

        Args:
            others_only: Indicates wheter to include self-linkages.

        Returns:
            self_connectedness: A (n_nodes * 1) vector with connectedness values.
        """
        self_connectedness = np.diag(self.adjacency_matrix).reshape(-1, 1)
        return self_connectedness

    def total_connectedness(
        self,
        others_only: bool = True,
    ) -> np.ndarray:
        """Get the total links of each node (incoming and outgoing).

        Args:
            others_only: Indicates wheter to include self-linkages.

        Returns:
            total_connectedness: A (n_nodes * 1) vector with connectedness values.
        """
        total_connectedness = self.in_connectedness(
            others_only=others_only
        ) + self.out_connectedness(others_only=others_only)
        if not others_only:
            total_connectedness -= self.self_connectedness()
        return total_connectedness

    def net_connectedness(
        self,
    ) -> np.ndarray:
        """Get the differnece of links of each node (outgoing less incoming).

        Returns:
            net_connectedness: A (n_nodes * 1) vector with connectedness values.
        """
        net_connectedness = self.out_connectedness(
            others_only=True
        ) - self.in_connectedness(others_only=True)
        return net_connectedness

    def amplification_factor(
        self,
        others_only: bool = False,
    ) -> np.ndarray:
        """Get the amplification factors of each node.

        amplification factor = (out_connectedness + self_connectedness)
            / (in_connectedness + self_connectedness)

        The inclusion of self_connectedness can be turned off with others_only=True.

        Args:
            others_only: Indicates wheter to include self-linkages.

        Returns:
            amplification_factor: A (n_nodes * 1) vector with amplifier values.
        """
        amplification_factor = self.out_connectedness(
            others_only=others_only
        ) / self.in_connectedness(others_only=others_only)
        return amplification_factor

    def absorption_rate(
        self,
        others_only: bool = False,
    ) -> np.ndarray:
        """Get the absorption rates of each node.

        absorption rate = self_connectedness / (in_connectedness + self_connectedness)

        The inclusion of self_connectedness in the denominator can be turned
        off with others_only=True.

        Args:
            others_only: Indicates wheter to include self-linkages in denominator.

        Returns:
            absorption_rate: A (n_nodes * 1) vector with absorption rates.
        """
        absorption_rate = self.self_connectedness() / self.in_connectedness(
            others_only=others_only
        )
        return absorption_rate

    def average_connectedness(
        self,
        others_only: bool = True,
    ) -> float:
        """Get the total links of each node (incoming and outgoing).

        Args:
            others_only: Indicates wheter to include self-linkages.

        Returns:
            average_connectedness: Average connectedness value in the table.
        """
        in_connectedness = self.in_connectedness(others_only=others_only)
        average_connectedness = in_connectedness.mean()
        return average_connectedness

    def in_concentration(
        self,
        others_only: bool = True,
        measure: str = "herfindahl_index",
    ) -> np.ndarray:
        """Calculate the concentration of incoming links per node (row-wise).

        Args:
            others_only: Indicates wheter to include self-linkages.
            measure: One of 'power_law_exponent', 'herfindahl_index', 'entropy'.

        Returns:
            in_concentration: A (n_nodes * 1) vector with concentration values.
        """
        if measure not in ["power_law_exponent", "herfindahl_index", "entropy"]:
            raise ValueError(
                "measure neets to be one of 'power_law_exponent', 'herfindahl_index', 'entropy'"
            )
        table = self.adjacency_matrix.copy()
        n = self.n_nodes

        # remove diagonal values
        if others_only:
            table = table[~np.eye(n, dtype=bool)].reshape(n, n - 1)

        # scale rows to one
        table /= table.sum(axis=1).reshape(n, 1)

        # calculate concentration
        if measure == "power_law_exponent":
            in_concentration = power_law_exponent(
                table,
                axis=1,
                invert=True,
            )
        elif measure == "herfindahl_index":
            in_concentration = herfindahl_index(table, axis=1)
        elif measure == "entropy":
            in_concentration = sp.stats.entropy(table, axis=1, base=n - others_only)
        in_concentration = in_concentration.reshape(-1, 1)

        return in_concentration

    def out_concentration(
        self,
        others_only: bool = True,
        measure: str = "herfindahl_index",
    ) -> np.ndarray:
        """Calculate the concentration of outgoing links per node (column-wise).

        Args:
            others_only: Indicates wheter to include self-linkages.
            measure: One of 'power_law_exponent', 'herfindahl_index', 'entropy'.

        Returns:
            out_concentration: A (n_nodes * 1) vector with concentration values.
        """
        if measure not in ["power_law_exponent", "herfindahl_index", "entropy"]:
            raise ValueError(
                "measure neets to be one of 'power_law_exponent', 'herfindahl_index', 'entropy'"
            )
        table = self.adjacency_matrix
        n = self.n_nodes

        # remove diagonal values
        if others_only:
            table = table[~np.eye(n, dtype=bool)].reshape(n - 1, n)

        # scale columns to one
        table /= table.sum(axis=0).reshape(1, n)

        # calculate concentration
        if measure == "power_law_exponent":
            out_concentration = power_law_exponent(
                table,
                axis=0,
                invert=True,
            )
        elif measure == "herfindahl_index":
            out_concentration = herfindahl_index(table, axis=0)
        elif measure == "entropy":
            out_concentration = sp.stats.entropy(table, axis=0, base=n - others_only)
        out_concentration = out_concentration.reshape(-1, 1)

        return out_concentration

    def to_graph(
        self,
    ) -> nx.classes.digraph.DiGraph:
        """Create a networkx Graph object from a connectedness table.

        Returns:
            graph: The connectedness table as a networkx DiGraph object.
        """
        graph = nx.convert_matrix.from_numpy_array(
            self.adjacency_matrix, create_using=nx.DiGraph
        )
        return graph.reverse()

    def in_eigenvector_centrality(
        self,
    ) -> np.ndarray:
        """Calculate the eigenvector centrality of incoming links per node (row-wise).

        Returns:
            in_centrality: A (n_nodes * 1) vector with centrality values.
        """
        # compute the largest right eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(self.adjacency_matrix)
        idx = np.argmax(eigenvalues)
        in_page_rank = eigenvectors[:, idx].flatten().real
        in_page_rank *= np.sign(in_page_rank.sum())

        return in_page_rank.reshape(-1, 1)

    def out_eigenvector_centrality(self) -> np.ndarray:
        """Calculate the eigenvector centrality of outgoing links per node (column-wise).

        Returns:
            out_centrality: A (n_nodes * 1) vector with centrality values.
        """
        # compute the largest left eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(self.adjacency_matrix.T)
        idx = np.argmax(eigenvalues)
        out_page_rank = eigenvectors[:, idx].flatten().real
        out_page_rank *= np.sign(out_page_rank.sum())

        return out_page_rank.reshape(-1, 1)

    def in_page_rank(
        self,
        alpha: float = 0.85,
        weights: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate the page rank of incoming links per node (row-wise).

        Args:
            alpha: Damping parameter for PageRank, default=0.85.
            weights: Probabilities of starting in a node (personalization).

        Returns:
            in_rank: A (n_nodes * 1) vector with centrality values.
        """
        # retrieve inputs
        table = self.adjacency_matrix
        out_connectedness = self.out_connectedness(others_only=False)

        # normalize weights
        if weights is None:
            weights = np.ones([self.n_nodes, 1])
        weights /= weights.sum()

        # calculate google matrix
        google_matrix = alpha * table @ np.linalg.inv(
            np.diag(out_connectedness.squeeze())
        ) + (1 - alpha) * weights @ np.ones([1, self.n_nodes])

        # compute the largest right eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(google_matrix)
        idx = np.argmax(eigenvalues)
        in_page_rank = eigenvectors[:, idx].flatten().real
        in_page_rank *= np.sign(in_page_rank.sum())

        return in_page_rank.reshape(-1, 1)

    def out_page_rank(
        self,
        alpha: float = 0.85,
        weights: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate the page rank of incoming links per node (row-wise).

        Args:
            alpha: Damping parameter for PageRank, default=0.85.
            weights: Probabilities of starting in a node (personalization).

        Returns:
            in_rank: A (n_nodes * 1) vector with centrality values.
        """
        # retrieve inputs
        table = self.adjacency_matrix
        in_connectedness = self.in_connectedness(others_only=False)

        # normalize weights
        if weights is None:
            weights = np.ones([self.n_nodes, 1])
        weights /= weights.sum()

        # calculate google matrix
        google_matrix = (
            alpha * np.linalg.inv(np.diag(in_connectedness.squeeze())) @ table
            + (1 - alpha) * np.ones([self.n_nodes, 1]) @ weights.T
        )

        # compute the largest left eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(google_matrix.T)
        idx = np.argmax(eigenvalues)
        out_page_rank = eigenvectors[:, idx].flatten().real
        out_page_rank *= np.sign(out_page_rank.sum())

        return out_page_rank.reshape(-1, 1)
