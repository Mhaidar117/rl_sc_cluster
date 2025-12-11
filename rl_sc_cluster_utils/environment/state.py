"""State extraction for the clustering environment."""

from typing import Dict, List, Optional

from anndata import AnnData
import numpy as np
from scipy.stats import entropy

from .cache import ClusteringCache
from .utils import (
    compute_clustering_quality_metrics,
    compute_enrichment_scores,
    compute_gag_enrichment_metrics,
    get_embeddings,
)


class StateExtractor:
    """
    Extract 35-dimensional state vector from AnnData clustering state.

    State Components (35 dimensions):
    - Global metrics (3): n_clusters, mean_size, entropy
    - Quality metrics (3): silhouette, modularity, balance
    - GAG enrichment (28): 7 gene sets × 4 metrics each
    - Progress (1): step / max_steps

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data
    gene_sets : dict
        Dictionary mapping gene set names to lists of gene names
    normalize : bool, optional
        Whether to normalize state vector (default: False)
    """

    def __init__(
        self,
        adata: AnnData,
        gene_sets: Optional[Dict[str, List[str]]] = None,
        normalize: bool = False,
    ) -> None:
        self.adata = adata
        self.gene_sets = gene_sets or {}
        self.normalize = normalize

        # Cache permanent data
        self._embeddings = None
        self._neighbors_computed = False

        # Validate and cache embeddings using shared utility
        self._embeddings = get_embeddings(adata)

        # Check if neighbors graph exists
        if "neighbors" in adata.uns:
            self._neighbors_computed = True

        # Initialize cache for clustering-dependent metrics
        self._cache = ClusteringCache(max_size=100)

        # Normalization ranges (will be computed if normalize=True)
        self._state_min = None
        self._state_max = None

    def extract_state(self, adata: AnnData, step: int, max_steps: int) -> np.ndarray:
        """
        Extract 35-dimensional state vector from current clustering.

        Parameters
        ----------
        adata : AnnData
            Annotated data object with current clustering
        step : int
            Current step in episode
        max_steps : int
            Maximum steps in episode

        Returns
        -------
        state : np.ndarray
            35-dimensional state vector
        """
        # Initialize state vector
        state = np.zeros(35, dtype=np.float64)

        # Extract each component
        global_metrics = self._compute_global_metrics(adata)
        quality_metrics = self._compute_quality_metrics(adata)
        gag_enrichment = self._compute_gag_enrichment(adata)
        progress = step / max_steps

        # Assemble state vector
        state[0:3] = global_metrics
        state[3:6] = quality_metrics
        state[6:34] = gag_enrichment
        state[34] = progress

        # Normalize if enabled
        if self.normalize:
            state = self._normalize_state(state)

        return state

    def _compute_global_metrics(self, adata: AnnData) -> np.ndarray:
        """
        Compute global clustering metrics.

        Returns
        -------
        metrics : np.ndarray
            [n_clusters/n_cells, mean_size/n_cells, entropy]
        """
        metrics = np.zeros(3, dtype=np.float64)

        # Get cluster labels
        if "clusters" not in adata.obs:
            # No clustering yet - return zeros
            return metrics

        cluster_labels = adata.obs["clusters"]
        n_cells = adata.n_obs
        n_clusters = len(cluster_labels.unique())

        # Metric 0: Normalized cluster count
        metrics[0] = n_clusters / n_cells

        # Metric 1: Normalized mean cluster size
        cluster_sizes = cluster_labels.value_counts()
        mean_size = cluster_sizes.mean()
        metrics[1] = mean_size / n_cells

        # Metric 2: Cluster size entropy
        if n_clusters > 1:
            p = cluster_sizes.values / cluster_sizes.sum()
            metrics[2] = entropy(p)
        else:
            metrics[2] = 0.0

        return metrics

    def _compute_quality_metrics(self, adata: AnnData) -> np.ndarray:
        """
        Compute clustering quality metrics using shared utilities.

        Returns
        -------
        metrics : np.ndarray
            [silhouette, modularity, balance]
        """
        # Check cache first (only if clusters exist)
        if "clusters" in adata.obs:
            cluster_labels = adata.obs["clusters"]
            cached = self._cache.get(cluster_labels, "quality_metrics")
            if cached is not None:
                return cached["metrics"]
        else:
            cluster_labels = None

        # Use shared utility function
        silhouette, modularity, balance = compute_clustering_quality_metrics(
            adata,
            self._embeddings,
            neighbors_computed=self._neighbors_computed,
            cluster_key="clusters",
        )

        metrics = np.array([silhouette, modularity, balance], dtype=np.float64)

        # Cache result (only if clusters exist)
        if cluster_labels is not None:
            self._cache.set(cluster_labels, "quality_metrics", {"metrics": metrics})

        return metrics

    def _compute_gag_enrichment(self, adata: AnnData) -> np.ndarray:
        """
        Compute GAG enrichment metrics for all gene sets using shared utilities.

        Returns
        -------
        metrics : np.ndarray
            28-dimensional vector (7 gene sets × 4 metrics)
        """
        metrics = np.zeros(28, dtype=np.float64)

        # Get cluster labels
        if "clusters" not in adata.obs:
            return metrics

        # If no gene sets provided, return zeros
        if not self.gene_sets:
            return metrics

        # Ensure we have exactly 7 gene sets (pad with empty if needed)
        gene_set_names = list(self.gene_sets.keys())[:7]
        while len(gene_set_names) < 7:
            gene_set_names.append(f"empty_set_{len(gene_set_names)}")

        # Check cache first
        cluster_labels = adata.obs["clusters"]
        cached = self._cache.get(cluster_labels, "gag_enrichment")
        if cached is not None:
            return cached["metrics"]

        # Compute metrics using shared utility
        gag_metrics = compute_gag_enrichment_metrics(adata, self.gene_sets, cluster_key="clusters")

        # Assemble into state vector format
        for i, gene_set_name in enumerate(gene_set_names):
            base_idx = i * 4

            if gene_set_name in gag_metrics:
                set_metrics = gag_metrics[gene_set_name]
                metrics[base_idx + 0] = set_metrics["mean"]
                metrics[base_idx + 1] = set_metrics["max"]
                metrics[base_idx + 2] = set_metrics["f_stat"]
                metrics[base_idx + 3] = set_metrics["mi"]

        # Cache result
        self._cache.set(cluster_labels, "gag_enrichment", {"metrics": metrics})

        return metrics

    def _compute_enrichment_scores(
        self, adata: AnnData, gene_set: List[str]
    ) -> Optional[np.ndarray]:
        """
        Compute enrichment scores for a gene set (simplified AUCell).

        This method is kept for backward compatibility but delegates to
        the shared utility function.

        Parameters
        ----------
        adata : AnnData
            Annotated data object
        gene_set : list
            List of gene names

        Returns
        -------
        scores : np.ndarray or None
            Enrichment scores for each cell, or None if gene set invalid
        """
        return compute_enrichment_scores(adata, gene_set)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state vector to [0, 1] range.

        Parameters
        ----------
        state : np.ndarray
            Raw state vector

        Returns
        -------
        normalized_state : np.ndarray
            Normalized state vector
        """
        # Initialize normalization ranges if not set
        if self._state_min is None:
            # Define expected ranges for each component
            self._state_min = np.array(
                [
                    # Global metrics
                    0.0,
                    0.0,
                    0.0,  # n_clusters, mean_size, entropy (all >= 0)
                    # Quality metrics
                    -1.0,
                    -0.5,
                    0.0,  # silhouette, modularity, balance
                    # GAG enrichment (28 values)
                    *([0.0] * 28),  # All enrichment metrics >= 0
                    # Progress
                    0.0,
                ]
            )

            self._state_max = np.array(
                [
                    # Global metrics
                    1.0,
                    1.0,
                    5.0,  # n_clusters, mean_size, entropy (log scale)
                    # Quality metrics
                    1.0,
                    1.0,
                    1.0,  # silhouette, modularity, balance
                    # GAG enrichment
                    *([1.0, 1.0, 10.0, 3.0] * 7),  # mean, max, F-stat, MI for 7 sets
                    # Progress
                    1.0,
                ]
            )

        # Normalize
        normalized = (state - self._state_min) / (self._state_max - self._state_min + 1e-10)

        # Clip to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized
