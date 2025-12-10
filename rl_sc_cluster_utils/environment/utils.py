"""Utility functions for the clustering environment."""

from typing import Dict, List, Optional, Tuple

from anndata import AnnData
import numpy as np
import scanpy as sc
from scipy.stats import entropy, f_oneway
from sklearn.metrics import mutual_info_score, silhouette_score


def validate_adata(adata: AnnData) -> None:
    """
    Validate AnnData object has required fields.

    Required fields:
    - .obsm['X_scvi'] or .X: Embedding matrix or expression data
    - .uns['neighbors']: k-NN graph (optional, computed if missing)
    - .obs['clusters']: Cluster labels (optional, computed at reset)

    Parameters
    ----------
    adata : AnnData
        Annotated data object to validate

    Raises
    ------
    ValueError
        If required fields are missing
    """
    # Check for embeddings or expression data
    has_embeddings = "X_scvi" in adata.obsm
    has_expression = adata.X is not None

    if not has_embeddings and not has_expression:
        raise ValueError("AnnData must contain either obsm['X_scvi'] or X matrix")


def get_embeddings(adata: AnnData) -> np.ndarray:
    """
    Get embeddings from AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object

    Returns
    -------
    embeddings : np.ndarray
        Embedding matrix (n_cells x n_features)

    Raises
    ------
    ValueError
        If no embeddings or expression data available
    """
    if "X_scvi" in adata.obsm:
        return adata.obsm["X_scvi"]
    elif adata.X is not None:
        return adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    else:
        raise ValueError("AnnData must contain either obsm['X_scvi'] or X matrix")


def compute_clustering_quality_metrics(
    adata: AnnData,
    embeddings: np.ndarray,
    neighbors_computed: bool = False,
    cluster_key: str = "clusters",
) -> Tuple[float, float, float]:
    """
    Compute clustering quality metrics: silhouette, modularity, and balance.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with cluster labels
    embeddings : np.ndarray
        Embedding matrix for silhouette computation
    neighbors_computed : bool, optional
        Whether neighbors graph exists (default: False)
    cluster_key : str, optional
        Key for cluster labels in adata.obs (default: "clusters")

    Returns
    -------
    silhouette : float
        Silhouette score [-1, 1], 0 if single cluster
    modularity : float
        Graph modularity [-0.5, 1], 0 if single cluster or no neighbors
    balance : float
        Cluster balance [0, 1], 1 if single cluster

    Examples
    --------
    >>> silhouette, modularity, balance = compute_clustering_quality_metrics(
    ...     adata, embeddings, neighbors_computed=True
    ... )
    """
    # Default values
    silhouette_val = 0.0
    modularity_val = 0.0
    balance_val = 0.0  # Default to 0 when no clusters

    # Check if cluster labels exist
    if cluster_key not in adata.obs:
        return silhouette_val, modularity_val, balance_val

    cluster_labels = adata.obs[cluster_key]
    n_clusters = len(cluster_labels.unique())
    n_cells = adata.n_obs

    # Silhouette score
    if n_clusters > 1 and n_clusters < n_cells:
        try:
            silhouette_val = silhouette_score(embeddings, cluster_labels)
        except Exception:
            silhouette_val = 0.0

    # Graph modularity
    if neighbors_computed and n_clusters > 1:
        try:
            modularity_val = sc.metrics.clustering.modularity(
                adata,
                label_key=cluster_key,
                use_rep="X_scvi" if "X_scvi" in adata.obsm else None,
            )
        except Exception:
            modularity_val = 0.0

    # Cluster balance
    if n_clusters > 1:
        cluster_sizes = cluster_labels.value_counts()
        mean_size = cluster_sizes.mean()
        std_size = cluster_sizes.std()
        balance_val = 1.0 - (std_size / (mean_size + 1e-10))
    else:
        balance_val = 1.0  # Single cluster is perfectly balanced

    return silhouette_val, modularity_val, balance_val


def compute_enrichment_scores(
    adata: AnnData,
    gene_set: List[str],
) -> Optional[np.ndarray]:
    """
    Compute enrichment scores for a gene set (simplified AUCell).

    Computes mean expression of gene set per cell as a simplified
    enrichment score.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    gene_set : list
        List of gene names

    Returns
    -------
    scores : np.ndarray or None
        Enrichment scores for each cell (n_cells,), or None if gene set invalid

    Examples
    --------
    >>> scores = compute_enrichment_scores(adata, ['GENE1', 'GENE2'])
    >>> if scores is not None:
    ...     print(f"Mean enrichment: {scores.mean():.4f}")
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


def compute_gag_enrichment_metrics(
    adata: AnnData,
    gene_sets: Dict[str, List[str]],
    cluster_key: str = "clusters",
) -> Dict[str, Dict[str, float]]:
    """
    Compute GAG enrichment metrics for all gene sets.

    For each gene set, computes:
    - mean: Mean enrichment across clusters
    - max: Maximum cluster enrichment
    - f_stat: ANOVA F-statistic (enrichment ~ cluster)
    - mi: Mutual information (cluster, enrichment)

    Parameters
    ----------
    adata : AnnData
        Annotated data object with cluster labels
    gene_sets : dict
        Dictionary mapping gene set names to lists of gene names
    cluster_key : str, optional
        Key for cluster labels in adata.obs (default: "clusters")

    Returns
    -------
    metrics : dict
        Dictionary mapping gene set names to metric dictionaries.
        Each metric dict contains: mean, max, f_stat, mi

    Examples
    --------
    >>> gene_sets = {'set1': ['GENE1', 'GENE2'], 'set2': ['GENE3']}
    >>> metrics = compute_gag_enrichment_metrics(adata, gene_sets)
    >>> print(metrics['set1']['f_stat'])
    """
    metrics: Dict[str, Dict[str, float]] = {}

    # Check if cluster labels exist
    if cluster_key not in adata.obs:
        return metrics

    cluster_labels = adata.obs[cluster_key]
    n_clusters = len(cluster_labels.unique())

    # If no gene sets provided, return empty
    if not gene_sets:
        return metrics

    # Compute metrics for each gene set
    for gene_set_name, gene_set in gene_sets.items():
        set_metrics: Dict[str, float] = {
            "mean": 0.0,
            "max": 0.0,
            "f_stat": 0.0,
            "mi": 0.0,
        }

        # Skip if gene set is empty
        if not gene_set:
            metrics[gene_set_name] = set_metrics
            continue

        # Compute enrichment scores
        enrichment_scores = compute_enrichment_scores(adata, gene_set)

        if enrichment_scores is None:
            metrics[gene_set_name] = set_metrics
            continue

        # Compute per-cluster means
        cluster_means = []
        for cluster in cluster_labels.unique():
            cluster_mask = cluster_labels == cluster
            cluster_mean = enrichment_scores[cluster_mask].mean()
            cluster_means.append(cluster_mean)

        set_metrics["mean"] = np.mean(cluster_means)
        set_metrics["max"] = np.max(cluster_means)

        # ANOVA F-statistic
        if n_clusters > 1:
            try:
                groups = [
                    enrichment_scores[cluster_labels == cluster]
                    for cluster in cluster_labels.unique()
                ]
                f_stat, _ = f_oneway(*groups)
                set_metrics["f_stat"] = f_stat if not np.isnan(f_stat) else 0.0
            except Exception:
                set_metrics["f_stat"] = 0.0

        # Mutual information
        if n_clusters > 1:
            try:
                # Bin enrichment scores into 10 bins
                enrichment_bins = np.digitize(
                    enrichment_scores,
                    bins=np.linspace(enrichment_scores.min(), enrichment_scores.max(), 11),
                )
                mi = mutual_info_score(cluster_labels, enrichment_bins)
                set_metrics["mi"] = mi if not np.isnan(mi) else 0.0
            except Exception:
                set_metrics["mi"] = 0.0

        metrics[gene_set_name] = set_metrics

    return metrics


def compute_global_clustering_metrics(
    adata: AnnData,
    cluster_key: str = "clusters",
) -> Tuple[int, float, float, int]:
    """
    Compute global clustering metrics.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with cluster labels
    cluster_key : str, optional
        Key for cluster labels in adata.obs (default: "clusters")

    Returns
    -------
    n_clusters : int
        Number of clusters
    mean_size : float
        Mean cluster size
    size_entropy : float
        Entropy of cluster size distribution
    n_singletons : int
        Number of singleton clusters (< 10 cells)

    Examples
    --------
    >>> n_clusters, mean_size, entropy, n_singletons = compute_global_clustering_metrics(adata)
    """
    # Default values
    if cluster_key not in adata.obs:
        return 0, 0.0, 0.0, 0

    cluster_labels = adata.obs[cluster_key]
    cluster_sizes = cluster_labels.value_counts()

    n_clusters = len(cluster_labels.unique())
    mean_size = cluster_sizes.mean()

    # Cluster size entropy
    if n_clusters > 1:
        p = cluster_sizes.values / cluster_sizes.sum()
        size_entropy = entropy(p)
    else:
        size_entropy = 0.0

    # Count singleton clusters (< 10 cells)
    n_singletons = (cluster_sizes < 10).sum()

    return n_clusters, mean_size, size_entropy, n_singletons
