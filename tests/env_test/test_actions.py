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
