"""Reward calculation for the clustering environment."""

from typing import Any, Dict, List, Optional, Tuple

from anndata import AnnData
import numpy as np

from .utils import (
    compute_clustering_quality_metrics,
    compute_gag_enrichment_metrics,
    compute_global_clustering_metrics,
    get_embeddings,
)


class RewardCalculator:
    """
    Compute composite reward for clustering environment.

    Reward formula:
        R = α·Q_cluster + β·Q_GAG - δ·Penalty

    Where:
    - Q_cluster: Clustering quality (silhouette, modularity, balance)
    - Q_GAG: GAG enrichment separation (normalized F-statistics)
    - Penalty: Degenerate states, singletons, resolution bounds

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data
    gene_sets : dict
        Dictionary mapping gene set names to lists of gene names
    alpha : float, optional
        Weight for clustering quality (default: 0.6)
    beta : float, optional
        Weight for GAG enrichment (default: 0.4)
    delta : float, optional
        Weight for penalties (default: 1.0)

    Examples
    --------
    >>> calculator = RewardCalculator(adata, gene_sets)
    >>> reward, info = calculator.compute_reward(adata, resolution_clamped=False)
    >>> print(f"Reward: {reward:.4f}, Q_cluster: {info['Q_cluster']:.4f}")
    """

    def __init__(
        self,
        adata: AnnData,
        gene_sets: Optional[Dict[str, List[str]]] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        delta: float = 1.0,
    ) -> None:
        self.adata = adata
        self.gene_sets = gene_sets or {}
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

        # Cache embeddings
        self._embeddings = get_embeddings(adata)

        # Check if neighbors graph exists
        self._neighbors_computed = "neighbors" in adata.uns

    def compute_reward(
        self,
        adata: AnnData,
        previous_reward: Optional[float] = None,
        resolution_clamped: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute composite reward for current clustering state.

        Parameters
        ----------
        adata : AnnData
            Annotated data object with current clustering
        previous_reward : float, optional
            Previous reward value (unused, kept for potential future use)
        resolution_clamped : bool, optional
            Whether resolution was clamped in last action (default: False)

        Returns
        -------
        reward : float
            Composite reward value
        info : dict
            Dictionary containing reward components:
            - Q_cluster: Clustering quality score
            - Q_GAG: GAG enrichment score
            - penalty: Total penalty
            - silhouette: Silhouette score
            - modularity: Graph modularity
            - balance: Cluster balance
            - n_clusters: Number of clusters
            - n_singletons: Number of singleton clusters
            - mean_f_stat: Mean F-statistic across gene sets

        Examples
        --------
        >>> reward, info = calculator.compute_reward(adata)
        >>> print(f"Reward: {reward:.4f}")
        """
        # Compute Q_cluster
        Q_cluster, q_cluster_info = self._compute_q_cluster(adata)

        # Compute Q_GAG
        Q_GAG, q_gag_info = self._compute_q_gag(adata)

        # Compute penalty
        penalty, penalty_info = self._compute_penalty(adata, resolution_clamped)

        # Composite reward
        reward = self.alpha * Q_cluster + self.beta * Q_GAG - self.delta * penalty

        # Build info dict
        info = {
            "reward": reward,
            "Q_cluster": Q_cluster,
            "Q_GAG": Q_GAG,
            "penalty": penalty,
            **q_cluster_info,
            **q_gag_info,
            **penalty_info,
        }

        return reward, info

    def _compute_q_cluster(self, adata: AnnData) -> Tuple[float, Dict[str, float]]:
        """
        Compute clustering quality score.

        Formula: Q_cluster = 0.5 * silhouette + 0.3 * modularity + 0.2 * balance

        Parameters
        ----------
        adata : AnnData
            Annotated data object with cluster labels

        Returns
        -------
        Q_cluster : float
            Clustering quality score
        info : dict
            Individual component values
        """
        # Get quality metrics from shared utility
        silhouette, modularity, balance = compute_clustering_quality_metrics(
            adata,
            self._embeddings,
            neighbors_computed=self._neighbors_computed,
            cluster_key="clusters",
        )

        # Weighted combination
        Q_cluster = 0.5 * silhouette + 0.3 * modularity + 0.2 * balance

        info = {
            "silhouette": silhouette,
            "modularity": modularity,
            "balance": balance,
        }

        return Q_cluster, info

    def _compute_q_gag(self, adata: AnnData) -> Tuple[float, Dict[str, Any]]:
        """
        Compute GAG enrichment score.

        Uses normalized F-statistics: Q_GAG = mean(log1p(f_stat) / 10.0) across gene sets

        Parameters
        ----------
        adata : AnnData
            Annotated data object with cluster labels

        Returns
        -------
        Q_GAG : float
            GAG enrichment score
        info : dict
            F-statistics per gene set and mean
        """
        # If no gene sets, return 0
        if not self.gene_sets:
            return 0.0, {"mean_f_stat": 0.0, "f_stats": {}}

        # Get GAG metrics from shared utility
        gag_metrics = compute_gag_enrichment_metrics(adata, self.gene_sets, cluster_key="clusters")

        # Extract F-statistics
        f_stats = {}
        f_stats_normalized = []

        for set_name, metrics in gag_metrics.items():
            f_stat = metrics["f_stat"]
            f_stats[set_name] = f_stat

            # Normalize F-stat: log1p(f_stat) / 10.0 to scale to ~[0, 1]
            f_norm = np.log1p(f_stat) / 10.0
            f_stats_normalized.append(f_norm)

        # Average normalized F-statistics
        if f_stats_normalized:
            Q_GAG = np.mean(f_stats_normalized)
            mean_f_stat = np.mean(list(f_stats.values()))
        else:
            Q_GAG = 0.0
            mean_f_stat = 0.0

        info = {
            "mean_f_stat": mean_f_stat,
            "f_stats": f_stats,
        }

        return Q_GAG, info

    def _compute_penalty(
        self, adata: AnnData, resolution_clamped: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute penalty for degenerate states.

        Penalty components:
        - +1.0 if n_clusters == 1 (too few clusters)
        - +1.0 if n_clusters > 0.3 * n_cells (too many clusters)
        - +0.1 per singleton cluster (< 10 cells)
        - +0.1 if resolution was clamped

        Parameters
        ----------
        adata : AnnData
            Annotated data object with cluster labels
        resolution_clamped : bool, optional
            Whether resolution was clamped in last action

        Returns
        -------
        penalty : float
            Total penalty value
        info : dict
            Penalty breakdown
        """
        penalty = 0.0
        n_cells = adata.n_obs

        # Get global metrics
        n_clusters, mean_size, size_entropy, n_singletons = compute_global_clustering_metrics(
            adata, cluster_key="clusters"
        )

        # Penalty for too few clusters
        if n_clusters == 1:
            penalty += 1.0

        # Penalty for too many clusters
        if n_clusters > 0.3 * n_cells:
            penalty += 1.0

        # Penalty for singleton clusters (< 10 cells)
        singleton_penalty = n_singletons * 0.1
        penalty += singleton_penalty

        # Penalty for resolution clamping
        bounds_penalty = 0.1 if resolution_clamped else 0.0
        penalty += bounds_penalty

        info = {
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
            "singleton_penalty": singleton_penalty,
            "bounds_penalty": bounds_penalty,
        }

        return penalty, info


class RewardNormalizer:
    """
    Normalize rewards using running statistics.

    Keeps track of reward history and computes running mean/std
    for normalization. This can help stabilize RL training.

    Parameters
    ----------
    clip_range : float, optional
        Clip normalized rewards to [-clip_range, clip_range] (default: 10.0)

    Examples
    --------
    >>> normalizer = RewardNormalizer()
    >>> normalizer.update(reward)
    >>> normalized = normalizer.normalize(reward)
    """

    def __init__(self, clip_range: float = 10.0) -> None:
        self.clip_range = clip_range
        self._rewards: List[float] = []
        self._mean: float = 0.0
        self._std: float = 1.0

    @property
    def mean(self) -> float:
        """Get running mean."""
        return self._mean

    @property
    def std(self) -> float:
        """Get running standard deviation."""
        return self._std

    @property
    def count(self) -> int:
        """Get number of rewards observed."""
        return len(self._rewards)

    def update(self, reward: float) -> None:
        """
        Update running statistics with new reward.

        Parameters
        ----------
        reward : float
            Reward value to add to history
        """
        self._rewards.append(reward)

        # Update running statistics
        if len(self._rewards) >= 2:
            self._mean = np.mean(self._rewards)
            self._std = np.std(self._rewards)
        elif len(self._rewards) == 1:
            self._mean = reward
            self._std = 1.0  # Default std for single value

    def normalize(self, reward: float) -> float:
        """
        Normalize reward using running statistics.

        Formula: normalized = (reward - mean) / (std + epsilon)

        Parameters
        ----------
        reward : float
            Reward value to normalize

        Returns
        -------
        normalized : float
            Normalized reward, clipped to [-clip_range, clip_range]
        """
        # Normalize
        normalized = (reward - self._mean) / (self._std + 1e-10)

        # Clip to range
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return float(normalized)

    def update_and_normalize(self, reward: float) -> float:
        """
        Update statistics and return normalized reward.

        Convenience method that combines update() and normalize().

        Parameters
        ----------
        reward : float
            Reward value to process

        Returns
        -------
        normalized : float
            Normalized reward
        """
        self.update(reward)
        return self.normalize(reward)

    def reset(self) -> None:
        """
        Reset all statistics.

        Clears reward history and resets mean/std to defaults.
        """
        self._rewards = []
        self._mean = 0.0
        self._std = 1.0

    def get_stats(self) -> Dict[str, float]:
        """
        Get current statistics.

        Returns
        -------
        stats : dict
            Dictionary containing mean, std, and count
        """
        return {
            "mean": self._mean,
            "std": self._std,
            "count": len(self._rewards),
        }
