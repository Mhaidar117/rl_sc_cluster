"""Unit tests for ActionExecutor."""

import numpy as np
import pytest
from anndata import AnnData
import scanpy as sc

from rl_sc_cluster_utils.environment.actions import ActionExecutor, convert_cluster_ids_to_numeric


@pytest.fixture
def mock_adata():
    """Create a realistic mock AnnData object for testing."""
    n_obs = 100
    n_vars = 50

    # Create expression matrix
    X = np.random.randn(n_obs, n_vars)
    adata = AnnData(X=X)

    # Add embeddings (scVI-like)
    adata.obsm["X_scvi"] = np.random.randn(n_obs, 10)

    # Add gene names
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    # Compute neighbors graph
    sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=15)

    # Initial clustering
    sc.tl.leiden(adata, resolution=0.5, key_added="clusters", flavor="igraph")

    return adata


@pytest.fixture
def action_executor(mock_adata):
    """Create an ActionExecutor instance for testing."""
    return ActionExecutor(mock_adata)


def test_convert_cluster_ids_to_numeric(mock_adata):
    """Test cluster ID conversion to numeric."""
    # Ensure clusters are string type
    mock_adata.obs["clusters"] = mock_adata.obs["clusters"].astype(str)

    convert_cluster_ids_to_numeric(mock_adata)

    # Check that clusters are now numeric
    assert mock_adata.obs["clusters"].dtype in [np.int8, np.int16, np.int32, np.int64]


def test_action_executor_initialization(mock_adata):
    """Test ActionExecutor initialization."""
    executor = ActionExecutor(mock_adata, min_resolution=0.1, max_resolution=2.0)

    assert executor.adata is mock_adata
    assert executor.min_resolution == 0.1
    assert executor.max_resolution == 2.0
    assert executor._embeddings is not None


def test_action_executor_initialization_no_embeddings():
    """Test ActionExecutor initialization without embeddings."""
    adata = AnnData(X=np.random.randn(50, 20))
    # No X_scvi, should use X
    executor = ActionExecutor(adata)
    assert executor._embeddings is not None


def test_action_executor_initialization_error():
    """Test ActionExecutor initialization with invalid data."""
    adata = AnnData()
    # No X or X_scvi
    with pytest.raises(ValueError, match="AnnData must contain"):
        ActionExecutor(adata)


def test_validate_action_split_single_cluster(mock_adata):
    """Test validation of split action with only 1 cluster."""
    # Force single cluster
    mock_adata.obs["clusters"] = 0

    executor = ActionExecutor(mock_adata)
    is_valid, error_msg = executor._validate_action(0, 0.5)

    assert not is_valid
    assert "Cannot split" in error_msg


def test_validate_action_split_singletons(mock_adata):
    """Test validation of split action with all singletons."""
    # Create all singleton clusters
    mock_adata.obs["clusters"] = range(len(mock_adata))

    executor = ActionExecutor(mock_adata)
    is_valid, error_msg = executor._validate_action(0, 0.5)

    assert not is_valid
    assert "all clusters are singletons" in error_msg


def test_validate_action_merge_single_cluster(mock_adata):
    """Test validation of merge action with only 1 cluster."""
    # Force single cluster
    mock_adata.obs["clusters"] = 0

    executor = ActionExecutor(mock_adata)
    is_valid, error_msg = executor._validate_action(1, 0.5)

    assert not is_valid
    assert "Cannot merge" in error_msg


def test_validate_action_recluster_valid(mock_adata):
    """Test validation of re-cluster actions (always valid)."""
    executor = ActionExecutor(mock_adata)

    is_valid, error_msg = executor._validate_action(2, 0.5)
    assert is_valid
    assert error_msg is None

    is_valid, error_msg = executor._validate_action(3, 0.5)
    assert is_valid
    assert error_msg is None


def test_increment_resolution(action_executor):
    """Test increment resolution action."""
    result = action_executor._increment_resolution(0.5)

    assert result["success"] is True
    assert result["new_resolution"] == 0.6
    assert result["resolution_clamped"] is False
    assert result["no_change"] is False

    # Check that clustering was updated
    assert "clusters" in action_executor.adata.obs
    assert len(action_executor.adata.obs["clusters"].unique()) > 0


def test_increment_resolution_clamped(action_executor):
    """Test increment resolution action with clamping."""
    result = action_executor._increment_resolution(1.95)

    assert result["success"] is True
    assert result["new_resolution"] == 2.0
    assert result["resolution_clamped"] is True


def test_decrement_resolution(action_executor):
    """Test decrement resolution action."""
    result = action_executor._decrement_resolution(0.5)

    assert result["success"] is True
    assert result["new_resolution"] == 0.4
    assert result["resolution_clamped"] is False
    assert result["no_change"] is False


def test_decrement_resolution_clamped(action_executor):
    """Test decrement resolution action with clamping."""
    result = action_executor._decrement_resolution(0.15)

    assert result["success"] is True
    assert result["new_resolution"] == 0.1
    assert result["resolution_clamped"] is True


def test_merge_closest_pair(action_executor):
    """Test merge closest pair action."""
    initial_n_clusters = len(action_executor.adata.obs["clusters"].unique())

    result = action_executor._merge_closest_pair()

    assert result["success"] is True
    assert result["no_change"] is False

    # Should have one fewer cluster
    final_n_clusters = len(action_executor.adata.obs["clusters"].unique())
    assert final_n_clusters == initial_n_clusters - 1


def test_merge_closest_pair_single_cluster(mock_adata):
    """Test merge action with only 1 cluster."""
    # Force single cluster
    mock_adata.obs["clusters"] = 0

    executor = ActionExecutor(mock_adata)
    result = executor._merge_closest_pair()

    assert result["success"] is False
    assert result["no_change"] is True
    assert "Cannot merge" in result["error"]


def test_split_worst_cluster(action_executor):
    """Test split worst cluster action."""
    initial_n_clusters = len(action_executor.adata.obs["clusters"].unique())

    # Only test if we have clusters to split
    if initial_n_clusters > 1:
        result = action_executor._split_worst_cluster(0.5)

        # Should succeed (may or may not actually split)
        assert result["success"] is True

        if not result["no_change"]:
            # If it split, should have more clusters
            final_n_clusters = len(action_executor.adata.obs["clusters"].unique())
            assert final_n_clusters >= initial_n_clusters


def test_split_worst_cluster_single_cluster(mock_adata):
    """Test split action with only 1 cluster."""
    # Force single cluster
    mock_adata.obs["clusters"] = 0
    convert_cluster_ids_to_numeric(mock_adata)

    executor = ActionExecutor(mock_adata)
    # Should fail validation before trying to split
    is_valid, error_msg = executor._validate_action(0, 0.5)
    assert not is_valid
    assert "Cannot split" in error_msg

    # If we bypass validation, should fail because worst_cluster will be None
    # (no clusters with size >= 2)
    result = executor._split_worst_cluster(0.5)
    # Should fail because no valid cluster to split
    assert result["success"] is False
    assert result["no_change"] is True


def test_split_worst_cluster_singleton(mock_adata):
    """Test split action with singleton cluster."""
    # Create singleton clusters
    mock_adata.obs["clusters"] = range(len(mock_adata))

    executor = ActionExecutor(mock_adata)
    # Should fail validation
    is_valid, error_msg = executor._validate_action(0, 0.5)
    assert not is_valid
    assert "all clusters are singletons" in error_msg

    # If we bypass validation, should still fail gracefully
    result = executor._split_worst_cluster(0.5)
    # May fail due to validation or other reasons
    assert result["success"] is False or result["no_change"] is True


def test_execute_action_accept(action_executor):
    """Test execute action 4 (accept)."""
    result = action_executor.execute(4, 0.5)

    assert result["success"] is True
    assert result["no_change"] is False
    assert result["new_resolution"] == 0.5


def test_execute_action_invalid_split(action_executor):
    """Test execute invalid split action."""
    # Force single cluster
    action_executor.adata.obs["clusters"] = 0

    result = action_executor.execute(0, 0.5)

    assert result["success"] is False
    assert result["no_change"] is True
    assert result["error"] is not None


def test_execute_action_invalid_merge(action_executor):
    """Test execute invalid merge action."""
    # Force single cluster
    action_executor.adata.obs["clusters"] = 0

    result = action_executor.execute(1, 0.5)

    assert result["success"] is False
    assert result["no_change"] is True
    assert result["error"] is not None


def test_execute_action_recluster(action_executor):
    """Test execute re-cluster actions."""
    result = action_executor.execute(2, 0.5)

    assert result["success"] is True
    assert result["new_resolution"] == 0.6

    result = action_executor.execute(3, 0.6)

    assert result["success"] is True
    assert result["new_resolution"] == 0.5


def test_cluster_ids_numeric_after_action(action_executor):
    """Test that cluster IDs are numeric after action execution."""
    # Execute an action
    action_executor.execute(2, 0.5)

    # Check cluster IDs are numeric
    assert action_executor.adata.obs["clusters"].dtype in [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ]


def test_compute_centroids(action_executor):
    """Test centroid computation."""
    # Ensure clusters exist
    if "clusters" not in action_executor.adata.obs:
        pytest.skip("No clusters in adata")

    centroids = action_executor._compute_centroids()

    assert isinstance(centroids, dict)
    assert len(centroids) > 0

    # Check centroids have correct shape
    for cluster_id, centroid in centroids.items():
        # Cluster ID may be int or converted from string, check numeric
        assert isinstance(cluster_id, (int, np.integer, str, np.str_))
        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (action_executor._embeddings.shape[1],)


def test_find_closest_clusters(action_executor):
    """Test finding closest cluster pair."""
    centroids = action_executor._compute_centroids()

    if len(centroids) >= 2:
        closest_pair = action_executor._find_closest_clusters(centroids)

        assert closest_pair is not None
        assert isinstance(closest_pair, tuple)
        assert len(closest_pair) == 2
        assert closest_pair[0] in centroids
        assert closest_pair[1] in centroids


def test_find_worst_cluster(action_executor):
    """Test finding worst cluster by silhouette."""
    worst_cluster = action_executor._find_worst_cluster()

    # May return None if no valid clusters
    if worst_cluster is not None:
        assert worst_cluster in action_executor.adata.obs["clusters"].unique()


def test_action_preserves_adata_structure(action_executor):
    """Test that actions preserve AnnData structure."""
    original_n_obs = action_executor.adata.n_obs
    original_n_vars = action_executor.adata.n_vars

    # Execute an action
    action_executor.execute(2, 0.5)

    # Check structure preserved
    assert action_executor.adata.n_obs == original_n_obs
    assert action_executor.adata.n_vars == original_n_vars
    assert "clusters" in action_executor.adata.obs
    assert "neighbors" in action_executor.adata.uns


# ==============================================================================
# GAG-AWARE ACTION TESTS
# ==============================================================================


@pytest.fixture
def mock_adata_with_gene_sets():
    """Create a mock AnnData object with gene sets for GAG-aware testing."""
    n_obs = 100
    n_vars = 50

    # Create expression matrix
    X = np.random.randn(n_obs, n_vars)
    adata = AnnData(X=X)

    # Add embeddings (scVI-like)
    adata.obsm["X_scvi"] = np.random.randn(n_obs, 10)

    # Add gene names
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    # Compute neighbors graph
    sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=15)

    # Initial clustering
    sc.tl.leiden(adata, resolution=0.5, key_added="clusters", flavor="igraph")

    return adata


@pytest.fixture
def gene_sets(mock_adata_with_gene_sets):
    """Create gene sets using genes that exist in the mock data."""
    # Use first 5 genes for gene_set_1, next 5 for gene_set_2
    return {
        "gene_set_1": ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"],
        "gene_set_2": ["gene_5", "gene_6", "gene_7", "gene_8", "gene_9"],
    }


@pytest.fixture
def gag_aware_action_executor(mock_adata_with_gene_sets, gene_sets):
    """Create an ActionExecutor instance with gene_sets for GAG-aware testing."""
    return ActionExecutor(mock_adata_with_gene_sets, gene_sets=gene_sets)


def test_action_executor_with_gene_sets(mock_adata_with_gene_sets, gene_sets):
    """Test ActionExecutor initialization with gene_sets."""
    executor = ActionExecutor(mock_adata_with_gene_sets, gene_sets=gene_sets)

    assert executor.gene_sets == gene_sets
    assert len(executor.gene_sets) == 2


def test_action_executor_without_gene_sets(mock_adata):
    """Test ActionExecutor defaults to empty gene_sets."""
    executor = ActionExecutor(mock_adata)

    assert executor.gene_sets == {}


def test_find_highest_gag_heterogeneity_cluster(gag_aware_action_executor):
    """Test finding cluster with highest GAG heterogeneity."""
    cluster_id = gag_aware_action_executor._find_highest_gag_heterogeneity_cluster()

    # Should return a valid cluster or None
    if cluster_id is not None:
        assert cluster_id in gag_aware_action_executor.adata.obs["clusters"].unique()


def test_find_highest_gag_heterogeneity_cluster_no_gene_sets(mock_adata):
    """Test GAG heterogeneity returns None without gene_sets."""
    executor = ActionExecutor(mock_adata, gene_sets={})
    cluster_id = executor._find_highest_gag_heterogeneity_cluster()

    assert cluster_id is None


def test_find_lowest_silhouette_cluster(gag_aware_action_executor):
    """Test fallback silhouette-based cluster selection."""
    cluster_id = gag_aware_action_executor._find_lowest_silhouette_cluster()

    # Should return a valid cluster or None
    if cluster_id is not None:
        assert cluster_id in gag_aware_action_executor.adata.obs["clusters"].unique()


def test_find_worst_cluster_uses_gag_first(gag_aware_action_executor):
    """Test that _find_worst_cluster uses GAG heterogeneity when gene_sets available."""
    # Both methods should return valid clusters
    gag_cluster = gag_aware_action_executor._find_highest_gag_heterogeneity_cluster()
    worst_cluster = gag_aware_action_executor._find_worst_cluster()

    # If GAG cluster is available, worst_cluster should use it
    if gag_cluster is not None:
        assert worst_cluster == gag_cluster


def test_find_worst_cluster_fallback_to_silhouette(mock_adata):
    """Test that _find_worst_cluster falls back to silhouette without gene_sets."""
    executor = ActionExecutor(mock_adata, gene_sets={})
    worst_cluster = executor._find_worst_cluster()
    silhouette_cluster = executor._find_lowest_silhouette_cluster()

    # Without gene_sets, should use silhouette
    assert worst_cluster == silhouette_cluster


def test_find_most_similar_gag_clusters(gag_aware_action_executor):
    """Test finding cluster pair with most similar GAG profiles."""
    closest_pair = gag_aware_action_executor._find_most_similar_gag_clusters()

    # Should return a pair or None
    if closest_pair is not None:
        assert isinstance(closest_pair, tuple)
        assert len(closest_pair) == 2
        assert closest_pair[0] in gag_aware_action_executor.adata.obs["clusters"].unique()
        assert closest_pair[1] in gag_aware_action_executor.adata.obs["clusters"].unique()


def test_find_most_similar_gag_clusters_no_gene_sets(mock_adata):
    """Test GAG similarity returns None without gene_sets."""
    executor = ActionExecutor(mock_adata, gene_sets={})
    closest_pair = executor._find_most_similar_gag_clusters()

    assert closest_pair is None


def test_merge_closest_pair_uses_gag_first(gag_aware_action_executor):
    """Test that _merge_closest_pair uses GAG similarity when gene_sets available."""
    initial_n_clusters = len(gag_aware_action_executor.adata.obs["clusters"].unique())

    # Skip test if only 1 cluster (can't merge)
    if initial_n_clusters < 2:
        pytest.skip("Only 1 cluster in mock data, cannot test merge")

    # Execute merge
    result = gag_aware_action_executor._merge_closest_pair()

    # Should succeed
    assert result["success"] is True

    # Should have one fewer cluster
    if not result["no_change"]:
        final_n_clusters = len(gag_aware_action_executor.adata.obs["clusters"].unique())
        assert final_n_clusters == initial_n_clusters - 1


def test_merge_closest_pair_fallback_to_centroid(mock_adata):
    """Test that _merge_closest_pair falls back to centroid without gene_sets."""
    executor = ActionExecutor(mock_adata, gene_sets={})
    initial_n_clusters = len(executor.adata.obs["clusters"].unique())

    result = executor._merge_closest_pair()

    assert result["success"] is True
    if not result["no_change"]:
        final_n_clusters = len(executor.adata.obs["clusters"].unique())
        assert final_n_clusters == initial_n_clusters - 1


def test_split_worst_cluster_uses_gag(gag_aware_action_executor):
    """Test that split uses GAG heterogeneity when gene_sets available."""
    initial_n_clusters = len(gag_aware_action_executor.adata.obs["clusters"].unique())

    if initial_n_clusters > 1:
        result = gag_aware_action_executor._split_worst_cluster(0.5)
        assert result["success"] is True


def test_gag_aware_actions_with_empty_gene_set(mock_adata_with_gene_sets):
    """Test GAG-aware actions with empty gene set values."""
    gene_sets = {"empty_set": []}
    executor = ActionExecutor(mock_adata_with_gene_sets, gene_sets=gene_sets)

    # Should fall back to silhouette/centroid
    worst = executor._find_worst_cluster()
    assert worst is not None or len(executor.adata.obs["clusters"].unique()) == 1


def test_gag_aware_actions_with_invalid_genes(mock_adata_with_gene_sets):
    """Test GAG-aware actions with genes not in the data."""
    gene_sets = {"invalid_set": ["not_a_gene", "also_not_a_gene"]}
    executor = ActionExecutor(mock_adata_with_gene_sets, gene_sets=gene_sets)

    # Should fall back to silhouette/centroid
    gag_cluster = executor._find_highest_gag_heterogeneity_cluster()
    assert gag_cluster is None  # No valid enrichment scores

    # But fallback should work
    worst = executor._find_worst_cluster()
    # Either returns a cluster or None if single cluster
    assert worst is None or worst in executor.adata.obs["clusters"].unique()


def test_gag_heterogeneity_identifies_diverse_cluster(mock_adata_with_gene_sets, gene_sets):
    """Test that GAG heterogeneity correctly identifies diverse clusters."""
    # Create clusters with varying GAG heterogeneity
    # Cluster 0: low variance in gene expression
    # Cluster 1: high variance in gene expression (should be selected for split)

    n_cells = mock_adata_with_gene_sets.n_obs
    mid = n_cells // 2

    # Assign first half to cluster 0, second half to cluster 1
    mock_adata_with_gene_sets.obs["clusters"] = [0] * mid + [1] * (n_cells - mid)

    # Set gene_0 (part of gene_set_1) to have low variance in cluster 0
    # and high variance in cluster 1
    X = mock_adata_with_gene_sets.X.copy()

    # Cluster 0: uniform values
    X[:mid, 0] = 1.0

    # Cluster 1: high variance values
    X[mid:, 0] = np.random.randn(n_cells - mid) * 10  # High variance

    mock_adata_with_gene_sets.X = X

    executor = ActionExecutor(mock_adata_with_gene_sets, gene_sets=gene_sets)
    highest_heterogeneity_cluster = executor._find_highest_gag_heterogeneity_cluster()

    # Cluster 1 should have higher heterogeneity
    assert highest_heterogeneity_cluster == 1


def test_gag_similarity_identifies_similar_clusters(mock_adata_with_gene_sets, gene_sets):
    """Test that GAG similarity correctly identifies similar clusters."""
    n_cells = mock_adata_with_gene_sets.n_obs
    third = n_cells // 3

    # Create 3 clusters
    mock_adata_with_gene_sets.obs["clusters"] = (
        [0] * third + [1] * third + [2] * (n_cells - 2 * third)
    )

    X = mock_adata_with_gene_sets.X.copy()

    # Clusters 0 and 1: similar gene_set_1 expression
    X[:third, :5] = 5.0  # Cluster 0
    X[third : 2 * third, :5] = 5.0  # Cluster 1 (same as cluster 0)

    # Cluster 2: very different expression
    X[2 * third :, :5] = -5.0

    mock_adata_with_gene_sets.X = X

    executor = ActionExecutor(mock_adata_with_gene_sets, gene_sets=gene_sets)
    similar_pair = executor._find_most_similar_gag_clusters()

    # Clusters 0 and 1 should be most similar
    assert similar_pair is not None
    assert set(similar_pair) == {0, 1}
