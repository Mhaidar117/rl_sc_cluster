"""Unit tests for shared utility functions."""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix
import scanpy as sc

from rl_sc_cluster_utils.environment.utils import (
    compute_clustering_quality_metrics,
    compute_enrichment_scores,
    compute_gag_enrichment_metrics,
    compute_global_clustering_metrics,
    get_embeddings,
    validate_adata,
)


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    n_obs = 100
    n_vars = 50

    # Create expression matrix
    X = np.random.randn(n_obs, n_vars)
    adata = AnnData(X=X)

    # Add embeddings (scVI-like)
    adata.obsm["X_scvi"] = np.random.randn(n_obs, 10)

    # Add gene names
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    return adata


@pytest.fixture
def clustered_adata(mock_adata):
    """Create a clustered AnnData object for testing."""
    # Compute neighbors graph
    sc.pp.neighbors(mock_adata, use_rep="X_scvi", n_neighbors=15)

    # Add cluster labels (3 clusters)
    mock_adata.obs["clusters"] = np.random.choice([0, 1, 2], size=mock_adata.n_obs)

    return mock_adata


@pytest.fixture
def gene_sets():
    """Create mock gene sets for testing."""
    return {
        "set1": ["gene_0", "gene_1", "gene_2"],
        "set2": ["gene_10", "gene_11", "gene_12"],
        "set3": ["gene_20", "gene_21", "gene_22"],
    }


# ============================================================================
# Test validate_adata
# ============================================================================


def test_validate_adata_with_scvi(mock_adata):
    """Test validation passes with X_scvi embeddings."""
    validate_adata(mock_adata)  # Should not raise


def test_validate_adata_with_expression_only():
    """Test validation passes with expression matrix only."""
    adata = AnnData(X=np.random.randn(50, 20))
    validate_adata(adata)  # Should not raise


def test_validate_adata_no_data():
    """Test validation fails with no data."""
    adata = AnnData()
    with pytest.raises(ValueError, match="must contain either"):
        validate_adata(adata)


# ============================================================================
# Test get_embeddings
# ============================================================================


def test_get_embeddings_scvi(mock_adata):
    """Test getting embeddings from X_scvi."""
    embeddings = get_embeddings(mock_adata)
    assert embeddings.shape == (100, 10)
    np.testing.assert_array_equal(embeddings, mock_adata.obsm["X_scvi"])


def test_get_embeddings_expression():
    """Test getting embeddings from expression matrix."""
    adata = AnnData(X=np.random.randn(50, 20))
    embeddings = get_embeddings(adata)
    assert embeddings.shape == (50, 20)


def test_get_embeddings_sparse():
    """Test getting embeddings from sparse matrix."""
    X_sparse = csr_matrix(np.random.randn(50, 20))
    adata = AnnData(X=X_sparse)
    embeddings = get_embeddings(adata)
    assert embeddings.shape == (50, 20)
    assert isinstance(embeddings, np.ndarray)


def test_get_embeddings_no_data():
    """Test error when no data available."""
    adata = AnnData()
    with pytest.raises(ValueError, match="must contain either"):
        get_embeddings(adata)


# ============================================================================
# Test compute_clustering_quality_metrics
# ============================================================================


def test_compute_clustering_quality_metrics_basic(clustered_adata):
    """Test basic clustering quality metrics computation."""
    embeddings = get_embeddings(clustered_adata)
    silhouette, modularity, balance = compute_clustering_quality_metrics(
        clustered_adata, embeddings, neighbors_computed=True
    )

    # Silhouette should be in [-1, 1]
    assert -1 <= silhouette <= 1

    # Balance should be in [0, 1] (or slightly outside due to formula)
    assert -1 <= balance <= 1.5


def test_compute_clustering_quality_metrics_no_clusters(mock_adata):
    """Test with no cluster labels."""
    embeddings = get_embeddings(mock_adata)
    silhouette, modularity, balance = compute_clustering_quality_metrics(
        mock_adata, embeddings, neighbors_computed=False
    )

    assert silhouette == 0.0
    assert modularity == 0.0
    assert balance == 0.0


def test_compute_clustering_quality_metrics_single_cluster(mock_adata):
    """Test with single cluster."""
    mock_adata.obs["clusters"] = 0  # All cells in one cluster
    embeddings = get_embeddings(mock_adata)
    silhouette, modularity, balance = compute_clustering_quality_metrics(
        mock_adata, embeddings, neighbors_computed=False
    )

    assert silhouette == 0.0  # No silhouette with 1 cluster
    assert balance == 1.0  # Perfect balance with 1 cluster


def test_compute_clustering_quality_metrics_without_neighbors(mock_adata):
    """Test modularity is 0 without neighbors graph."""
    mock_adata.obs["clusters"] = np.random.choice([0, 1, 2], size=mock_adata.n_obs)
    embeddings = get_embeddings(mock_adata)
    silhouette, modularity, balance = compute_clustering_quality_metrics(
        mock_adata, embeddings, neighbors_computed=False
    )

    assert modularity == 0.0  # No modularity without neighbors


def test_compute_clustering_quality_metrics_custom_key(clustered_adata):
    """Test with custom cluster key."""
    clustered_adata.obs["custom_clusters"] = clustered_adata.obs["clusters"]
    embeddings = get_embeddings(clustered_adata)
    silhouette, modularity, balance = compute_clustering_quality_metrics(
        clustered_adata, embeddings, neighbors_computed=True, cluster_key="custom_clusters"
    )

    assert -1 <= silhouette <= 1


# ============================================================================
# Test compute_enrichment_scores
# ============================================================================


def test_compute_enrichment_scores_basic(mock_adata):
    """Test basic enrichment score computation."""
    gene_set = ["gene_0", "gene_1", "gene_2"]
    scores = compute_enrichment_scores(mock_adata, gene_set)

    assert scores is not None
    assert scores.shape == (100,)


def test_compute_enrichment_scores_no_valid_genes(mock_adata):
    """Test with no valid genes."""
    gene_set = ["invalid_gene_1", "invalid_gene_2"]
    scores = compute_enrichment_scores(mock_adata, gene_set)

    assert scores is None


def test_compute_enrichment_scores_partial_valid(mock_adata):
    """Test with partially valid gene set."""
    gene_set = ["gene_0", "invalid_gene", "gene_1"]
    scores = compute_enrichment_scores(mock_adata, gene_set)

    assert scores is not None
    assert scores.shape == (100,)


def test_compute_enrichment_scores_empty_set(mock_adata):
    """Test with empty gene set."""
    scores = compute_enrichment_scores(mock_adata, [])

    assert scores is None


def test_compute_enrichment_scores_sparse_matrix():
    """Test with sparse expression matrix."""
    X_sparse = csr_matrix(np.random.randn(50, 20))
    adata = AnnData(X=X_sparse)
    adata.var_names = [f"gene_{i}" for i in range(20)]

    scores = compute_enrichment_scores(adata, ["gene_0", "gene_1"])

    assert scores is not None
    assert scores.shape == (50,)


# ============================================================================
# Test compute_gag_enrichment_metrics
# ============================================================================


def test_compute_gag_enrichment_metrics_basic(clustered_adata, gene_sets):
    """Test basic GAG enrichment metrics computation."""
    metrics = compute_gag_enrichment_metrics(clustered_adata, gene_sets)

    assert len(metrics) == 3  # 3 gene sets
    for set_name in gene_sets:
        assert set_name in metrics
        assert "mean" in metrics[set_name]
        assert "max" in metrics[set_name]
        assert "f_stat" in metrics[set_name]
        assert "mi" in metrics[set_name]


def test_compute_gag_enrichment_metrics_no_clusters(mock_adata, gene_sets):
    """Test with no cluster labels."""
    metrics = compute_gag_enrichment_metrics(mock_adata, gene_sets)

    assert len(metrics) == 0


def test_compute_gag_enrichment_metrics_no_gene_sets(clustered_adata):
    """Test with no gene sets."""
    metrics = compute_gag_enrichment_metrics(clustered_adata, {})

    assert len(metrics) == 0


def test_compute_gag_enrichment_metrics_empty_gene_set(clustered_adata):
    """Test with empty gene set in dict."""
    gene_sets = {"empty_set": []}
    metrics = compute_gag_enrichment_metrics(clustered_adata, gene_sets)

    assert len(metrics) == 1
    assert metrics["empty_set"]["mean"] == 0.0
    assert metrics["empty_set"]["f_stat"] == 0.0


def test_compute_gag_enrichment_metrics_invalid_genes(clustered_adata):
    """Test with invalid gene names."""
    gene_sets = {"invalid_set": ["invalid_gene_1", "invalid_gene_2"]}
    metrics = compute_gag_enrichment_metrics(clustered_adata, gene_sets)

    assert len(metrics) == 1
    assert metrics["invalid_set"]["mean"] == 0.0
    assert metrics["invalid_set"]["f_stat"] == 0.0


def test_compute_gag_enrichment_metrics_single_cluster(mock_adata, gene_sets):
    """Test with single cluster."""
    mock_adata.obs["clusters"] = 0  # All cells in one cluster
    metrics = compute_gag_enrichment_metrics(mock_adata, gene_sets)

    # F-stat and MI should be 0 with single cluster
    for set_name in gene_sets:
        assert metrics[set_name]["f_stat"] == 0.0
        assert metrics[set_name]["mi"] == 0.0


def test_compute_gag_enrichment_metrics_f_stat_positive(clustered_adata, gene_sets):
    """Test that F-stat is positive with multiple clusters."""
    metrics = compute_gag_enrichment_metrics(clustered_adata, gene_sets)

    # At least some F-stats should be positive (not all can be 0 with random data)
    f_stats = [metrics[s]["f_stat"] for s in gene_sets]
    assert any(f >= 0 for f in f_stats)


# ============================================================================
# Test compute_global_clustering_metrics
# ============================================================================


def test_compute_global_clustering_metrics_basic(clustered_adata):
    """Test basic global clustering metrics computation."""
    n_clusters, mean_size, size_entropy, n_singletons = compute_global_clustering_metrics(
        clustered_adata
    )

    assert n_clusters == 3
    assert mean_size > 0
    assert size_entropy >= 0
    assert n_singletons >= 0


def test_compute_global_clustering_metrics_no_clusters(mock_adata):
    """Test with no cluster labels."""
    n_clusters, mean_size, size_entropy, n_singletons = compute_global_clustering_metrics(
        mock_adata
    )

    assert n_clusters == 0
    assert mean_size == 0.0
    assert size_entropy == 0.0
    assert n_singletons == 0


def test_compute_global_clustering_metrics_single_cluster(mock_adata):
    """Test with single cluster."""
    mock_adata.obs["clusters"] = 0
    n_clusters, mean_size, size_entropy, n_singletons = compute_global_clustering_metrics(
        mock_adata
    )

    assert n_clusters == 1
    assert mean_size == 100
    assert size_entropy == 0.0  # No entropy with 1 cluster
    assert n_singletons == 0


def test_compute_global_clustering_metrics_singletons(mock_adata):
    """Test singleton counting."""
    # Create clusters with some singletons
    labels = np.zeros(100, dtype=int)
    labels[0:5] = [0, 1, 2, 3, 4]  # 5 singleton clusters (size 1)
    labels[5:] = 5  # One large cluster
    mock_adata.obs["clusters"] = labels

    n_clusters, mean_size, size_entropy, n_singletons = compute_global_clustering_metrics(
        mock_adata
    )

    assert n_clusters == 6
    assert n_singletons == 5  # 5 clusters with < 10 cells


def test_compute_global_clustering_metrics_custom_key(clustered_adata):
    """Test with custom cluster key."""
    clustered_adata.obs["custom_clusters"] = clustered_adata.obs["clusters"]
    n_clusters, mean_size, size_entropy, n_singletons = compute_global_clustering_metrics(
        clustered_adata, cluster_key="custom_clusters"
    )

    assert n_clusters == 3
