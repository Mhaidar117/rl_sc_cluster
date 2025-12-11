"""State extraction for the clustering environment."""

from typing import Dict, List, Optional

from anndata import AnnData
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity
import numpy as np
import scanpy as sc
from scipy.stats import entropy, f_oneway
from sklearn.metrics import mutual_info_score, silhouette_score


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

        # Validate and cache embeddings
        if "X_scvi" in adata.obsm:
            self._embeddings = adata.obsm["X_scvi"]
        elif adata.X is not None:
            # Fallback to raw expression if no embeddings
            self._embeddings = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        else:
            raise ValueError("AnnData must contain either obsm['X_scvi'] or X matrix")

        # Check if neighbors graph exists
        if "neighbors" in adata.uns:
            self._neighbors_computed = True

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
        Compute clustering quality metrics.

        Returns
        -------
        metrics : np.ndarray
            [silhouette, modularity, balance]
        """
        metrics = np.zeros(3, dtype=np.float64)

        # Get cluster labels
        if "clusters" not in adata.obs:
            return metrics

        cluster_labels = adata.obs["clusters"]
        n_clusters = len(cluster_labels.unique())

        # Metric 0: Silhouette score
        if n_clusters > 1 and n_clusters < adata.n_obs:
            try:
                metrics[0] = silhouette_score(self._embeddings, cluster_labels)
            except Exception:
                # Handle edge cases (e.g., all same cluster)
                metrics[0] = 0.0
        else:
            metrics[0] = 0.0

        # Metric 1: Graph modularity
        if self._neighbors_computed and n_clusters > 1:
            try:
                metrics[1] = self._compute_graph_modularity(adata, "clusters")
            except Exception:
                metrics[1] = 0.0
        else:
            metrics[1] = 0.0

        # Metric 2: Cluster balance
        cluster_sizes = cluster_labels.value_counts()
        if n_clusters > 1:
            mean_size = cluster_sizes.mean()
            std_size = cluster_sizes.std()
            metrics[2] = 1.0 - (std_size / (mean_size + 1e-10))
        else:
            metrics[2] = 1.0  # Single cluster is perfectly balanced

        return metrics

    def _compute_graph_modularity(
        self, adata: AnnData, cluster_key: str, neighbors_key: str = "neighbors"
    ) -> float:
        """
        Compute modularity of a clustering using the kNN graph from Scanpy.

        Parameters
        ----------
        adata : AnnData
            Annotated data object with neighbors graph
        cluster_key : str
            Key in adata.obs containing cluster labels
        neighbors_key : str
            Key in adata.uns containing neighbors info

        Returns
        -------
        float
            Modularity score
        """
        # Get the kNN graph Scanpy built
        conn_key = adata.uns[neighbors_key]["connectivities_key"]
        A = adata.obsp[conn_key]  # sparse adjacency matrix

        # Build an undirected weighted graph
        G = nx.from_scipy_sparse_array(A)

        # Turn cluster labels into a list of communities
        # IMPORTANT: Use integer indices (0 to n-1) to match graph node indices
        labels = adata.obs[cluster_key].astype("category")

        # Create communities using integer indices, not string cell names
        communities = []
        for c in labels.cat.categories:
            # Get boolean mask for this cluster
            mask = (labels == c).values
            # Get integer indices where mask is True
            cell_indices = set(np.where(mask)[0])
            communities.append(cell_indices)

        return nx_modularity(G, communities, weight="weight")

    def _compute_gag_enrichment(self, adata: AnnData) -> np.ndarray:
        """
        Compute GAG enrichment metrics for all gene sets.

        Returns
        -------
        metrics : np.ndarray
            28-dimensional vector (7 gene sets × 4 metrics)
        """
        metrics = np.zeros(28, dtype=np.float64)

        # Get cluster labels
        if "clusters" not in adata.obs:
            return metrics

        cluster_labels = adata.obs["clusters"]
        n_clusters = len(cluster_labels.unique())

        # If no gene sets provided, return zeros
        if not self.gene_sets:
            return metrics

        # Ensure we have exactly 7 gene sets (pad with empty if needed)
        gene_set_names = list(self.gene_sets.keys())[:7]
        while len(gene_set_names) < 7:
            gene_set_names.append(f"empty_set_{len(gene_set_names)}")

        # Compute metrics for each gene set
        for i, gene_set_name in enumerate(gene_set_names):
            base_idx = i * 4

            # Get gene set
            if gene_set_name in self.gene_sets:
                gene_set = self.gene_sets[gene_set_name]
            else:
                gene_set = []

            # Skip if gene set is empty
            if not gene_set:
                continue

            # Compute enrichment scores (simplified AUCell-like approach)
            enrichment_scores = self._compute_enrichment_scores(adata, gene_set)

            if enrichment_scores is None:
                continue

            # Metric 0: Mean enrichment across clusters
            cluster_means = []
            for cluster in cluster_labels.unique():
                cluster_mask = cluster_labels == cluster
                cluster_mean = enrichment_scores[cluster_mask].mean()
                cluster_means.append(cluster_mean)

            metrics[base_idx + 0] = np.mean(cluster_means)

            # Metric 1: Max enrichment
            metrics[base_idx + 1] = np.max(cluster_means)

            # Metric 2: ANOVA F-statistic
            if n_clusters > 1:
                try:
                    groups = [
                        enrichment_scores[cluster_labels == cluster]
                        for cluster in cluster_labels.unique()
                    ]
                    f_stat, _ = f_oneway(*groups)
                    metrics[base_idx + 2] = f_stat if not np.isnan(f_stat) else 0.0
                except Exception:
                    metrics[base_idx + 2] = 0.0
            else:
                metrics[base_idx + 2] = 0.0

            # Metric 3: Mutual information
            if n_clusters > 1:
                try:
                    # Bin enrichment scores into 10 bins
                    enrichment_bins = np.digitize(
                        enrichment_scores,
                        bins=np.linspace(enrichment_scores.min(), enrichment_scores.max(), 11),
                    )
                    mi = mutual_info_score(cluster_labels, enrichment_bins)
                    metrics[base_idx + 3] = mi if not np.isnan(mi) else 0.0
                except Exception:
                    metrics[base_idx + 3] = 0.0
            else:
                metrics[base_idx + 3] = 0.0

        return metrics

    def _compute_enrichment_scores(
        self, adata: AnnData, gene_set: List[str]
    ) -> Optional[np.ndarray]:
        """
        Compute enrichment scores for a gene set (simplified AUCell).

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
        # Find genes in the dataset
        valid_genes = [gene for gene in gene_set if gene in adata.var_names]

        if len(valid_genes) == 0:
            return None

        # Get expression matrix
        if isinstance(adata.X, np.ndarray):
            expr = adata.X
        else:
            expr = adata.X.toarray()

        # Get gene indices
        gene_indices = [adata.var_names.get_loc(gene) for gene in valid_genes]

        # Compute mean expression of gene set per cell (simplified enrichment)
        enrichment_scores = expr[:, gene_indices].mean(axis=1)

        # Flatten if needed
        if len(enrichment_scores.shape) > 1:
            enrichment_scores = enrichment_scores.flatten()

        return enrichment_scores

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
