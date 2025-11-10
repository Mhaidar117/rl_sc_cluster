"""Unit tests for StateExtractor."""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from rl_sc_cluster_utils.environment.state import StateExtractor


@pytest.fixture
def realistic_adata():
    """Create a realistic AnnData object for testing."""
    n_obs = 100
    n_vars = 50

    # Create expression matrix
    X = np.random.randn(n_obs, n_vars)

    # Create AnnData
    adata = AnnData(X=X)

    # Add embeddings (scVI-like)
    adata.obsm["X_scvi"] = np.random.randn(n_obs, 10)

    # Add cluster labels (3 clusters)
    adata.obs["clusters"] = np.random.choice(["cluster_0", "cluster_1", "cluster_2"], size=n_obs)

    # Add gene names
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    # Add neighbors graph (mock)
    adata.uns["neighbors"] = {
        "connectivities": csr_matrix(np.random.rand(n_obs, n_obs) > 0.9),
        "distances": csr_matrix(np.random.rand(n_obs, n_obs)),
        "params": {"n_neighbors": 15, "method": "umap"},
    }

    return adata


@pytest.fixture
def gene_sets():
    """Create mock gene sets for testing."""
    return {
        "set_1": ["gene_0", "gene_1", "gene_2"],
        "set_2": ["gene_3", "gene_4", "gene_5"],
        "set_3": ["gene_6", "gene_7", "gene_8"],
        "set_4": ["gene_9", "gene_10", "gene_11"],
        "set_5": ["gene_12", "gene_13", "gene_14"],
        "set_6": ["gene_15", "gene_16", "gene_17"],
        "set_7": ["gene_18", "gene_19", "gene_20"],
    }


@pytest.fixture
def state_extractor(realistic_adata, gene_sets):
    """Create a StateExtractor instance for testing."""
    return StateExtractor(realistic_adata, gene_sets, normalize=False)


def test_initialization(realistic_adata, gene_sets):
    """Test StateExtractor initialization."""
    extractor = StateExtractor(realistic_adata, gene_sets, normalize=False)

    assert extractor.adata is realistic_adata
    assert extractor.gene_sets == gene_sets
    assert extractor.normalize is False
    assert extractor._embeddings is not None
    assert extractor._neighbors_computed is True


def test_initialization_no_embeddings():
    """Test initialization with no embeddings falls back to X."""
    adata = AnnData(X=np.random.randn(50, 20))
    extractor = StateExtractor(adata, {}, normalize=False)

    assert extractor._embeddings is not None
    assert extractor._embeddings.shape == (50, 20)


def test_initialization_no_data():
    """Test initialization fails with no data."""
    adata = AnnData()
    with pytest.raises(ValueError, match="AnnData must contain"):
        StateExtractor(adata, {}, normalize=False)


def test_extract_state_shape(state_extractor, realistic_adata):
    """Test that extract_state returns correct shape."""
    state = state_extractor.extract_state(realistic_adata, step=0, max_steps=15)

    assert state.shape == (35,)
    assert state.dtype == np.float64


def test_extract_state_components(state_extractor, realistic_adata):
    """Test that state components are in correct positions."""
    state = state_extractor.extract_state(realistic_adata, step=5, max_steps=15)

    # Check that all components are finite
    assert np.all(np.isfinite(state))

    # Check progress component
    assert state[34] == pytest.approx(5 / 15)


def test_compute_global_metrics(state_extractor, realistic_adata):
    """Test global metrics computation."""
    metrics = state_extractor._compute_global_metrics(realistic_adata)

    assert metrics.shape == (3,)
    assert np.all(np.isfinite(metrics))

    # Check ranges
    assert 0 <= metrics[0] <= 1  # n_clusters / n_cells
    assert 0 <= metrics[1] <= 1  # mean_size / n_cells
    assert metrics[2] >= 0  # entropy


def test_compute_global_metrics_no_clusters(state_extractor):
    """Test global metrics with no clustering."""
    adata = AnnData(X=np.random.randn(50, 20))
    metrics = state_extractor._compute_global_metrics(adata)

    assert metrics.shape == (3,)
    assert np.all(metrics == 0.0)


def test_compute_global_metrics_single_cluster(state_extractor, realistic_adata):
    """Test global metrics with single cluster."""
    realistic_adata.obs["clusters"] = "cluster_0"
    metrics = state_extractor._compute_global_metrics(realistic_adata)

    assert metrics[0] == pytest.approx(1 / realistic_adata.n_obs)  # 1 cluster
    assert metrics[1] == pytest.approx(1.0)  # All cells in one cluster
    assert metrics[2] == pytest.approx(0.0)  # No entropy


def test_compute_quality_metrics(state_extractor, realistic_adata):
    """Test quality metrics computation."""
    metrics = state_extractor._compute_quality_metrics(realistic_adata)

    assert metrics.shape == (3,)
    assert np.all(np.isfinite(metrics))

    # Check ranges
    assert -1 <= metrics[0] <= 1  # silhouette
    # modularity can be negative
    assert 0 <= metrics[2] <= 1  # balance


def test_compute_quality_metrics_no_clusters(state_extractor):
    """Test quality metrics with no clustering."""
    adata = AnnData(X=np.random.randn(50, 20))
    adata.obsm["X_scvi"] = np.random.randn(50, 10)
    extractor = StateExtractor(adata, {}, normalize=False)
    metrics = extractor._compute_quality_metrics(adata)

    assert metrics.shape == (3,)
    assert np.all(metrics == 0.0)


def test_compute_quality_metrics_single_cluster(state_extractor, realistic_adata):
    """Test quality metrics with single cluster."""
    realistic_adata.obs["clusters"] = "cluster_0"
    metrics = state_extractor._compute_quality_metrics(realistic_adata)

    assert metrics[0] == pytest.approx(0.0)  # No silhouette with 1 cluster
    assert metrics[2] == pytest.approx(1.0)  # Perfect balance with 1 cluster


def test_compute_gag_enrichment(state_extractor, realistic_adata):
    """Test GAG enrichment computation."""
    metrics = state_extractor._compute_gag_enrichment(realistic_adata)

    assert metrics.shape == (28,)
    assert np.all(np.isfinite(metrics))


def test_compute_gag_enrichment_no_gene_sets(realistic_adata):
    """Test GAG enrichment with no gene sets."""
    extractor = StateExtractor(realistic_adata, {}, normalize=False)
    metrics = extractor._compute_gag_enrichment(realistic_adata)

    assert metrics.shape == (28,)
    assert np.all(metrics == 0.0)


def test_compute_gag_enrichment_no_clusters(state_extractor):
    """Test GAG enrichment with no clustering."""
    adata = AnnData(X=np.random.randn(50, 20))
    adata.var_names = [f"gene_{i}" for i in range(20)]
    metrics = state_extractor._compute_gag_enrichment(adata)

    assert metrics.shape == (28,)
    assert np.all(metrics == 0.0)


def test_compute_enrichment_scores(state_extractor, realistic_adata):
    """Test enrichment score computation."""
    gene_set = ["gene_0", "gene_1", "gene_2"]
    scores = state_extractor._compute_enrichment_scores(realistic_adata, gene_set)

    assert scores is not None
    assert scores.shape == (realistic_adata.n_obs,)
    assert np.all(np.isfinite(scores))


def test_compute_enrichment_scores_invalid_genes(state_extractor, realistic_adata):
    """Test enrichment scores with invalid genes."""
    gene_set = ["invalid_gene_1", "invalid_gene_2"]
    scores = state_extractor._compute_enrichment_scores(realistic_adata, gene_set)

    assert scores is None


def test_compute_enrichment_scores_mixed_genes(state_extractor, realistic_adata):
    """Test enrichment scores with mix of valid and invalid genes."""
    gene_set = ["gene_0", "invalid_gene", "gene_1"]
    scores = state_extractor._compute_enrichment_scores(realistic_adata, gene_set)

    assert scores is not None
    assert scores.shape == (realistic_adata.n_obs,)


def test_normalization(realistic_adata, gene_sets):
    """Test state normalization."""
    extractor = StateExtractor(realistic_adata, gene_sets, normalize=True)
    state = extractor.extract_state(realistic_adata, step=5, max_steps=15)

    # All values should be in [0, 1] after normalization
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)


def test_normalization_disabled(realistic_adata, gene_sets):
    """Test that normalization can be disabled."""
    extractor = StateExtractor(realistic_adata, gene_sets, normalize=False)
    state = extractor.extract_state(realistic_adata, step=5, max_steps=15)

    # Values may be outside [0, 1] without normalization
    # Just check they're finite
    assert np.all(np.isfinite(state))


def test_state_consistency(state_extractor, realistic_adata):
    """Test that extracting state twice gives same result."""
    state1 = state_extractor.extract_state(realistic_adata, step=0, max_steps=15)
    state2 = state_extractor.extract_state(realistic_adata, step=0, max_steps=15)

    np.testing.assert_array_equal(state1, state2)


def test_state_changes_with_step(state_extractor, realistic_adata):
    """Test that state changes with step (progress component)."""
    state1 = state_extractor.extract_state(realistic_adata, step=0, max_steps=15)
    state2 = state_extractor.extract_state(realistic_adata, step=5, max_steps=15)

    # Progress component should be different
    assert state1[34] != state2[34]
    assert state1[34] == pytest.approx(0.0)
    assert state2[34] == pytest.approx(5 / 15)


def test_edge_case_singleton_clusters(state_extractor, realistic_adata):
    """Test with singleton clusters (each cell is its own cluster)."""
    realistic_adata.obs["clusters"] = [f"cluster_{i}" for i in range(realistic_adata.n_obs)]

    state = state_extractor.extract_state(realistic_adata, step=0, max_steps=15)

    # Should not crash and return finite values
    assert state.shape == (35,)
    assert np.all(np.isfinite(state))


def test_edge_case_two_clusters(state_extractor, realistic_adata):
    """Test with exactly two clusters."""
    realistic_adata.obs["clusters"] = np.where(
        np.random.rand(realistic_adata.n_obs) > 0.5, "cluster_0", "cluster_1"
    )

    state = state_extractor.extract_state(realistic_adata, step=0, max_steps=15)

    # Should not crash and return finite values
    assert state.shape == (35,)
    assert np.all(np.isfinite(state))


def test_sparse_expression_matrix(gene_sets):
    """Test with sparse expression matrix."""
    n_obs = 100
    n_vars = 50

    # Create sparse matrix
    X = csr_matrix(np.random.rand(n_obs, n_vars))

    adata = AnnData(X=X)
    adata.obsm["X_scvi"] = np.random.randn(n_obs, 10)
    adata.obs["clusters"] = np.random.choice(["cluster_0", "cluster_1"], size=n_obs)
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    extractor = StateExtractor(adata, gene_sets, normalize=False)
    state = extractor.extract_state(adata, step=0, max_steps=15)

    assert state.shape == (35,)
    assert np.all(np.isfinite(state))
