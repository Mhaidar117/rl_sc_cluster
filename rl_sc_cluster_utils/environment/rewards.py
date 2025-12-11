"""Reward calculation for the clustering environment."""

from typing import Any, Dict, List, Optional, Tuple

from anndata import AnnData
import numpy as np

from .cache import ClusteringCache
from .utils import (
    compute_clustering_quality_metrics,
    compute_enrichment_scores,
    compute_gag_enrichment_metrics,
    compute_global_clustering_metrics,
    get_embeddings,
)


class RewardCalculator:
    """
    Compute composite reward for clustering environment.

    Supports three reward modes:
    - "absolute": R = α·Q_cluster + β·Q_GAG_transformed - δ·Penalty
    - "improvement": R = (current_potential - previous_potential) + exploration_bonus
    - "shaped": R = raw_reward - baseline + offset (keeps rewards non-negative)

    Where:
    - Q_cluster: Clustering quality (silhouette, modularity, balance)
    - Q_GAG: GAG enrichment separation (normalized F-statistics)
    - Q_GAG_transformed: (Q_GAG * gag_scale)² if gag_nonlinear=True
    - Penalty: Degenerate states, singletons, resolution bounds

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data
    gene_sets : dict
        Dictionary mapping gene set names to lists of gene names
    alpha : float, optional
        Weight for clustering quality (default: 0.2)
    beta : float, optional
        Weight for GAG enrichment (default: 2.0)
    delta : float, optional
        Weight for penalties (default: 0.01)
    reward_mode : str, optional
        Reward mode: "absolute", "improvement", or "shaped" (default: "shaped")
    gag_nonlinear : bool, optional
        Apply non-linear transformation (Q_GAG * scale)² (default: True)
    gag_scale : float, optional
        Scaling factor for GAG transformation (default: 6.0)
    exploration_bonus : float, optional
        Bonus per step for improvement mode (default: 0.2)
    silhouette_shift : float, optional
        Shift amount to keep silhouette non-negative (default: 0.5)
    early_termination_penalty : float, optional
        Penalty for Accept action before minimum steps (default: -5.0)
    min_steps_before_accept : int, optional
        Minimum steps before Accept action is allowed without penalty (default: 10)

    Examples
    --------
    >>> calculator = RewardCalculator(adata, gene_sets, reward_mode="shaped")
    >>> reward, info = calculator.compute_reward(adata, action=0, current_step=5)
    >>> print(f"Reward: {reward:.4f}, Q_cluster: {info['Q_cluster']:.4f}")
    """

    def __init__(
        self,
        adata: AnnData,
        gene_sets: Optional[Dict[str, List[str]]] = None,
        alpha: float = 0.2,
        beta: float = 2.0,
        delta: float = 0.01,
        reward_mode: str = "shaped",
        gag_nonlinear: bool = True,
        gag_scale: float = 6.0,
        exploration_bonus: float = 0.2,
        silhouette_shift: float = 0.5,
        early_termination_penalty: float = -5.0,
        min_steps_before_accept: int = 10,
    ) -> None:
        self.adata = adata
        self.gene_sets = gene_sets or {}
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.reward_mode = reward_mode
        self.gag_nonlinear = gag_nonlinear
        self.gag_scale = gag_scale
        self.exploration_bonus = exploration_bonus
        self.silhouette_shift = silhouette_shift
        self.early_termination_penalty = early_termination_penalty
        self.min_steps_before_accept = min_steps_before_accept

        # Cache embeddings
        self._embeddings = get_embeddings(adata)

        # Check if neighbors graph exists
        self._neighbors_computed = "neighbors" in adata.uns

        # Precompute static enrichment scores (independent of clustering)
        self._precomputed_enrichment: Dict[str, np.ndarray] = {}
        for gene_set_name, gene_set in self.gene_sets.items():
            enrichment = compute_enrichment_scores(adata, gene_set)
            if enrichment is not None:
                self._precomputed_enrichment[gene_set_name] = enrichment

        # Initialize cache for clustering-dependent metrics
        self._cache = ClusteringCache(max_size=100)

        # Internal state for improvement and shaped modes
        self._previous_potential: Optional[float] = None
        self._reward_history: List[float] = []
        self._baseline: float = 0.0

    def compute_reward(
        self,
        adata: AnnData,
        previous_reward: Optional[float] = None,
        resolution_clamped: bool = False,
        action: Optional[int] = None,
        current_step: Optional[int] = None,
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
        action : int, optional
            Action taken (0-4). Required for early termination penalty.
        current_step : int, optional
            Current step in episode. Required for early termination penalty.

        Returns
        -------
        reward : float
            Composite reward value (mode-dependent)
        info : dict
            Dictionary containing reward components:
            - Q_cluster: Clustering quality score
            - Q_GAG: Raw GAG enrichment score (before transformation)
            - Q_GAG_transformed: Transformed GAG score (if gag_nonlinear=True)
            - penalty: Total penalty
            - silhouette: Raw silhouette score (preserved for interpretation)
            - silhouette_for_reward: Silhouette value used in reward computation
            - modularity: Graph modularity
            - balance: Cluster balance
            - n_clusters: Number of clusters
            - n_singletons: Number of singleton clusters
            - mean_f_stat: Mean F-statistic across gene sets
            - reward_mode: Reward mode used
            - baseline: Baseline value (for shaped mode)

        Examples
        --------
        >>> reward, info = calculator.compute_reward(adata, action=0, current_step=5)
        >>> print(f"Reward: {reward:.4f}")
        """
        # Compute Q_cluster (preserves raw silhouette)
        Q_cluster, q_cluster_info = self._compute_q_cluster(adata)

        # Compute Q_GAG (raw, before transformation)
        Q_GAG, q_gag_info = self._compute_q_gag(adata)

        # Apply GAG non-linear transformation if enabled
        if self.gag_nonlinear:
            Q_GAG_transformed = (Q_GAG * self.gag_scale) ** 2
        else:
            Q_GAG_transformed = Q_GAG

        # Compute penalty
        penalty, penalty_info = self._compute_penalty(adata, resolution_clamped)

        # Compute raw reward (before mode-specific shaping)
        raw_reward = self.alpha * Q_cluster + self.beta * Q_GAG_transformed - self.delta * penalty

        # Apply reward mode
        if self.reward_mode == "absolute":
            reward = raw_reward

        elif self.reward_mode == "improvement":
            # Delta reward: improvement in potential
            current_potential = raw_reward

            if self._previous_potential is None:
                # First step: use absolute potential + exploration bonus
                reward = current_potential + self.exploration_bonus
            else:
                # Subsequent steps: improvement + exploration bonus
                reward = (current_potential - self._previous_potential) + self.exploration_bonus

            self._previous_potential = current_potential

        elif self.reward_mode == "shaped":
            # Shaped reward: subtract baseline to keep rewards non-negative
            # Update baseline (running average)
            self._reward_history.append(raw_reward)
            if len(self._reward_history) > 100:  # Keep last 100 for baseline
                self._reward_history.pop(0)
            self._baseline = np.mean(self._reward_history) if self._reward_history else 0.0

            # Shaped reward: subtract baseline, add small positive offset
            reward = raw_reward - self._baseline + 0.1  # +0.1 ensures mostly positive

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        # Apply early termination penalty if applicable
        early_termination_penalty_applied = False
        if action == 4 and current_step is not None:
            if current_step < self.min_steps_before_accept:
                reward = self.early_termination_penalty
                early_termination_penalty_applied = True

        # Build info dict (always include raw values)
        # Note: "reward" is the return value, not included in info to avoid redundancy
        info = {
            "Q_cluster": Q_cluster,
            "Q_GAG": Q_GAG,  # Raw GAG (before transformation)
            "Q_GAG_transformed": Q_GAG_transformed if self.gag_nonlinear else Q_GAG,
            "penalty": penalty,
            "reward_mode": self.reward_mode,
            "baseline": self._baseline if self.reward_mode == "shaped" else None,
            "early_termination_penalty_applied": early_termination_penalty_applied,
            **q_cluster_info,  # Includes raw silhouette
            **q_gag_info,
            **penalty_info,
        }

        return reward, info

    def reset(self) -> None:
        """
        Reset internal state for new episode.

        Clears previous potential and baseline tracking.
        Also clears cache since clustering state changes on reset.
        """
        self._previous_potential = None
        # Keep baseline across episodes for shaped mode (or reset if desired)
        # self._reward_history = []
        # self._baseline = 0.0
        # Clear cache on reset since clustering state changes
        self._cache.clear()

    def _compute_q_cluster(self, adata: AnnData) -> Tuple[float, Dict[str, float]]:
        """
        Compute clustering quality score.

        Formula: Q_cluster = 0.5 * silhouette_for_reward + 0.3 * modularity + 0.2 * balance

        Always preserves raw silhouette in info dict for interpretation.
        Uses shifted silhouette for reward computation to avoid negative values.

        Parameters
        ----------
        adata : AnnData
            Annotated data object with cluster labels

        Returns
        -------
        Q_cluster : float
            Clustering quality score
        info : dict
            Individual component values, including raw silhouette
        """
        # Check cache first (only if clusters exist)
        if "clusters" in adata.obs:
            cluster_labels = adata.obs["clusters"]
            cached = self._cache.get(cluster_labels, "q_cluster")
            if cached is not None:
                return cached["Q_cluster"], cached["info"]
        else:
            cluster_labels = None

        # Get quality metrics from shared utility
        silhouette, modularity, balance = compute_clustering_quality_metrics(
            adata,
            self._embeddings,
            neighbors_computed=self._neighbors_computed,
            cluster_key="clusters",
        )

        # For reward: shift silhouette to keep non-negative (preserves relative differences)
        # This avoids negative rewards while preserving information
        silhouette_for_reward = max(0.0, silhouette + self.silhouette_shift)

        # Weighted combination
        Q_cluster = 0.5 * silhouette_for_reward + 0.3 * modularity + 0.2 * balance

        info = {
            "silhouette": silhouette,  # RAW silhouette (preserved!)
            "silhouette_for_reward": silhouette_for_reward,  # What was used in reward
            "modularity": modularity,
            "balance": balance,
        }

        # Cache result (only if clusters exist)
        if cluster_labels is not None:
            self._cache.set(cluster_labels, "q_cluster", {"Q_cluster": Q_cluster, "info": info})

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

        # Check cache first
        cluster_labels = adata.obs["clusters"]
        cached = self._cache.get(cluster_labels, "q_gag")
        if cached is not None:
            return cached["Q_GAG"], cached["info"]

        # Get GAG metrics from shared utility with precomputed enrichment scores
        gag_metrics = compute_gag_enrichment_metrics(
            adata,
            self.gene_sets,
            cluster_key="clusters",
            precomputed_enrichment=self._precomputed_enrichment,
        )

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

        # Cache result
        self._cache.set(cluster_labels, "q_gag", {"Q_GAG": Q_GAG, "info": info})

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

        Normalizes using statistics computed BEFORE adding the current reward,
        then updates statistics with the new reward. This ensures the first
        reward is normalized correctly (not to zero).

        Parameters
        ----------
        reward : float
            Reward value to process

        Returns
        -------
        normalized : float
            Normalized reward (using statistics before update)
        """
        # Normalize using current statistics (before adding this reward)
        normalized = self.normalize(reward)
        # Then update statistics with the new reward
        self.update(reward)
        return normalized

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
