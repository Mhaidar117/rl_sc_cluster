"""Unit tests for RewardCalculator and RewardNormalizer."""

import numpy as np
import pytest
from anndata import AnnData
import scanpy as sc

from rl_sc_cluster_utils.environment.rewards import RewardCalculator, RewardNormalizer


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


@pytest.fixture
def reward_calculator(clustered_adata, gene_sets):
    """Create a RewardCalculator instance for testing."""
    return RewardCalculator(clustered_adata, gene_sets)


# ============================================================================
# Test RewardCalculator Initialization
# ============================================================================


def test_reward_calculator_initialization(clustered_adata, gene_sets):
    """Test RewardCalculator initialization."""
    calculator = RewardCalculator(clustered_adata, gene_sets)

    # Updated defaults for GAG-focused reward
    assert calculator.alpha == 0.2
    assert calculator.beta == 2.0
    assert calculator.delta == 0.01
    assert calculator.reward_mode == "shaped"  # Default mode
    assert calculator.gag_nonlinear == True  # Default: apply GAG transformation
    assert calculator._embeddings is not None


def test_reward_calculator_custom_weights(clustered_adata, gene_sets):
    """Test RewardCalculator with custom weights."""
    calculator = RewardCalculator(clustered_adata, gene_sets, alpha=0.5, beta=0.3, delta=0.8)

    assert calculator.alpha == 0.5
    assert calculator.beta == 0.3
    assert calculator.delta == 0.8


def test_reward_calculator_no_gene_sets(clustered_adata):
    """Test RewardCalculator without gene sets."""
    calculator = RewardCalculator(clustered_adata, None)

    assert calculator.gene_sets == {}


# ============================================================================
# Test Q_cluster Computation
# ============================================================================


def test_compute_q_cluster_basic(reward_calculator, clustered_adata):
    """Test basic Q_cluster computation."""
    Q_cluster, info = reward_calculator._compute_q_cluster(clustered_adata)

    # Q_cluster should be a finite float
    assert np.isfinite(Q_cluster)

    # Info should contain components
    assert "silhouette" in info
    assert "modularity" in info
    assert "balance" in info

    # Components should be finite
    assert np.isfinite(info["silhouette"])
    assert np.isfinite(info["modularity"])
    assert np.isfinite(info["balance"])


def test_compute_q_cluster_formula(reward_calculator, clustered_adata):
    """Test Q_cluster formula: 0.5*silhouette_for_reward + 0.3*modularity + 0.2*balance."""
    Q_cluster, info = reward_calculator._compute_q_cluster(clustered_adata)

    # Q_cluster uses silhouette_for_reward (shifted), not raw silhouette
    expected = 0.5 * info["silhouette_for_reward"] + 0.3 * info["modularity"] + 0.2 * info["balance"]
    assert Q_cluster == pytest.approx(expected, rel=1e-6)


def test_compute_q_cluster_single_cluster(mock_adata, gene_sets):
    """Test Q_cluster with single cluster."""
    mock_adata.obs["clusters"] = 0  # All cells in one cluster
    calculator = RewardCalculator(mock_adata, gene_sets)

    Q_cluster, info = calculator._compute_q_cluster(mock_adata)

    # Silhouette should be 0 with single cluster
    assert info["silhouette"] == 0.0
    # Balance should be 1 with single cluster
    assert info["balance"] == 1.0


def test_compute_q_cluster_no_clusters(mock_adata, gene_sets):
    """Test Q_cluster with no cluster labels."""
    calculator = RewardCalculator(mock_adata, gene_sets)

    Q_cluster, info = calculator._compute_q_cluster(mock_adata)

    # Raw silhouette should be 0 with no clusters
    assert info["silhouette"] == 0.0
    assert info["modularity"] == 0.0
    assert info["balance"] == 0.0

    # silhouette_for_reward = max(0, 0 + 0.5) = 0.5 due to shift
    assert info["silhouette_for_reward"] == 0.5

    # Q_cluster = 0.5 * 0.5 + 0.3 * 0 + 0.2 * 0 = 0.25
    assert Q_cluster == pytest.approx(0.25)


# ============================================================================
# Test Q_GAG Computation
# ============================================================================


def test_compute_q_gag_basic(reward_calculator, clustered_adata):
    """Test basic Q_GAG computation."""
    Q_GAG, info = reward_calculator._compute_q_gag(clustered_adata)

    # Q_GAG should be a finite float
    assert np.isfinite(Q_GAG)

    # Info should contain f_stats
    assert "mean_f_stat" in info
    assert "f_stats" in info


def test_compute_q_gag_no_gene_sets(clustered_adata):
    """Test Q_GAG without gene sets."""
    calculator = RewardCalculator(clustered_adata, {})

    Q_GAG, info = calculator._compute_q_gag(clustered_adata)

    assert Q_GAG == 0.0
    assert info["mean_f_stat"] == 0.0


def test_compute_q_gag_single_cluster(mock_adata, gene_sets):
    """Test Q_GAG with single cluster."""
    mock_adata.obs["clusters"] = 0  # All cells in one cluster
    calculator = RewardCalculator(mock_adata, gene_sets)

    Q_GAG, info = calculator._compute_q_gag(mock_adata)

    # F-stats should be 0 with single cluster
    for set_name, f_stat in info["f_stats"].items():
        assert f_stat == 0.0


def test_compute_q_gag_normalization(reward_calculator, clustered_adata):
    """Test Q_GAG normalization formula: log1p(f_stat) / 10.0."""
    _, info = reward_calculator._compute_q_gag(clustered_adata)

    # Compute expected Q_GAG from F-stats
    f_stats = list(info["f_stats"].values())
    if f_stats:
        f_stats_normalized = [np.log1p(f) / 10.0 for f in f_stats]
        expected_Q_GAG = np.mean(f_stats_normalized)

        Q_GAG, _ = reward_calculator._compute_q_gag(clustered_adata)
        assert Q_GAG == pytest.approx(expected_Q_GAG, rel=1e-6)


# ============================================================================
# Test Penalty Computation
# ============================================================================


def test_compute_penalty_basic(reward_calculator, clustered_adata):
    """Test basic penalty computation."""
    penalty, info = reward_calculator._compute_penalty(clustered_adata)

    # Penalty should be a finite float
    assert np.isfinite(penalty)
    assert penalty >= 0.0

    # Info should contain breakdown
    assert "n_clusters" in info
    assert "n_singletons" in info
    assert "singleton_penalty" in info
    assert "bounds_penalty" in info


def test_compute_penalty_single_cluster(mock_adata, gene_sets):
    """Test penalty for single cluster (degenerate state)."""
    mock_adata.obs["clusters"] = 0  # All cells in one cluster
    calculator = RewardCalculator(mock_adata, gene_sets)

    penalty, info = calculator._compute_penalty(mock_adata)

    # Should have +1.0 penalty for single cluster
    assert penalty >= 1.0
    assert info["n_clusters"] == 1


def test_compute_penalty_too_many_clusters(mock_adata, gene_sets):
    """Test penalty for too many clusters (> 0.3 * n_cells)."""
    # Create 40 clusters for 100 cells (40% > 30% threshold)
    mock_adata.obs["clusters"] = np.arange(100) % 40
    calculator = RewardCalculator(mock_adata, gene_sets)

    penalty, info = calculator._compute_penalty(mock_adata)

    # Should have +1.0 penalty for too many clusters
    assert penalty >= 1.0
    assert info["n_clusters"] > 0.3 * mock_adata.n_obs


def test_compute_penalty_singletons(mock_adata, gene_sets):
    """Test penalty for singleton clusters."""
    # Create clusters with some singletons (< 10 cells)
    labels = np.zeros(100, dtype=int)
    labels[0:5] = [0, 1, 2, 3, 4]  # 5 singleton clusters (size 1)
    labels[5:] = 5  # One large cluster
    mock_adata.obs["clusters"] = labels
    calculator = RewardCalculator(mock_adata, gene_sets)

    penalty, info = calculator._compute_penalty(mock_adata)

    # Should have +0.1 penalty per singleton (5 singletons = 0.5)
    assert info["n_singletons"] == 5
    assert info["singleton_penalty"] == pytest.approx(0.5)


def test_compute_penalty_resolution_clamped(reward_calculator, clustered_adata):
    """Test penalty for resolution clamping."""
    penalty_unclamped, info_unclamped = reward_calculator._compute_penalty(
        clustered_adata, resolution_clamped=False
    )
    penalty_clamped, info_clamped = reward_calculator._compute_penalty(
        clustered_adata, resolution_clamped=True
    )

    # Clamped should have +0.1 additional penalty
    assert info_unclamped["bounds_penalty"] == 0.0
    assert info_clamped["bounds_penalty"] == 0.1
    assert penalty_clamped == pytest.approx(penalty_unclamped + 0.1)


# ============================================================================
# Test Composite Reward Computation
# ============================================================================


def test_compute_reward_basic(reward_calculator, clustered_adata):
    """Test basic reward computation."""
    reward, info = reward_calculator.compute_reward(clustered_adata)

    # Reward should be a finite float
    assert np.isfinite(reward)

    # Info should contain all components (reward is returned separately, not in info)
    assert "Q_cluster" in info
    assert "Q_GAG" in info
    assert "Q_GAG_transformed" in info  # New field
    assert "penalty" in info
    assert "reward_mode" in info  # New field


def test_compute_reward_formula(clustered_adata, gene_sets):
    """Test reward formula with absolute mode: R = α·Q_cluster + β·Q_GAG_transformed - δ·Penalty."""
    # Use absolute mode to directly test the formula
    calculator = RewardCalculator(
        clustered_adata, gene_sets,
        reward_mode="absolute",
        gag_nonlinear=True,
        gag_scale=6.0,
    )
    reward, info = calculator.compute_reward(clustered_adata)

    # Compute expected reward from components
    # With gag_nonlinear=True: Q_GAG_transformed = (Q_GAG * gag_scale)²
    expected = (
        calculator.alpha * info["Q_cluster"]
        + calculator.beta * info["Q_GAG_transformed"]
        - calculator.delta * info["penalty"]
    )
    assert reward == pytest.approx(expected, rel=1e-6)


def test_compute_reward_with_resolution_clamped(reward_calculator, clustered_adata):
    """Test reward computation with resolution clamped."""
    reward_unclamped, _ = reward_calculator.compute_reward(
        clustered_adata, resolution_clamped=False
    )
    reward_clamped, _ = reward_calculator.compute_reward(clustered_adata, resolution_clamped=True)

    # Clamped should have lower reward (additional penalty)
    assert reward_clamped < reward_unclamped


def test_compute_reward_info_completeness(reward_calculator, clustered_adata):
    """Test that reward info contains all expected fields."""
    _, info = reward_calculator.compute_reward(clustered_adata)

    # Note: "reward" is returned separately, not in info dict
    expected_fields = [
        "Q_cluster",
        "Q_GAG",
        "Q_GAG_transformed",  # New field
        "penalty",
        "silhouette",
        "silhouette_for_reward",  # New field
        "modularity",
        "balance",
        "mean_f_stat",
        "n_clusters",
        "n_singletons",
        "reward_mode",  # New field
    ]

    for field in expected_fields:
        assert field in info, f"Missing field: {field}"


# ============================================================================
# Test RewardNormalizer
# ============================================================================


def test_reward_normalizer_initialization():
    """Test RewardNormalizer initialization."""
    normalizer = RewardNormalizer()

    assert normalizer.mean == 0.0
    assert normalizer.std == 1.0
    assert normalizer.count == 0


def test_reward_normalizer_update():
    """Test RewardNormalizer update."""
    normalizer = RewardNormalizer()

    normalizer.update(1.0)
    assert normalizer.count == 1
    assert normalizer.mean == 1.0

    normalizer.update(3.0)
    assert normalizer.count == 2
    assert normalizer.mean == pytest.approx(2.0)


def test_reward_normalizer_normalize():
    """Test RewardNormalizer normalize."""
    normalizer = RewardNormalizer()

    # Add some values to get running stats
    normalizer.update(0.0)
    normalizer.update(2.0)
    normalizer.update(4.0)

    # Mean = 2.0, std = 1.63
    mean = normalizer.mean
    std = normalizer.std

    # Test normalization
    normalized = normalizer.normalize(2.0)
    expected = (2.0 - mean) / (std + 1e-10)
    assert normalized == pytest.approx(expected)


def test_reward_normalizer_clip():
    """Test RewardNormalizer clips to range."""
    normalizer = RewardNormalizer(clip_range=5.0)

    # Add values
    normalizer.update(0.0)
    normalizer.update(1.0)

    # Large value should be clipped
    normalized = normalizer.normalize(100.0)
    assert normalized <= 5.0
    assert normalized >= -5.0


def test_reward_normalizer_reset():
    """Test RewardNormalizer reset."""
    normalizer = RewardNormalizer()

    # Add some values
    normalizer.update(1.0)
    normalizer.update(2.0)
    normalizer.update(3.0)

    # Reset
    normalizer.reset()

    assert normalizer.count == 0
    assert normalizer.mean == 0.0
    assert normalizer.std == 1.0


def test_reward_normalizer_update_and_normalize():
    """Test RewardNormalizer update_and_normalize convenience method."""
    normalizer = RewardNormalizer()

    # First value
    normalized1 = normalizer.update_and_normalize(1.0)
    assert normalizer.count == 1

    # Second value
    normalized2 = normalizer.update_and_normalize(3.0)
    assert normalizer.count == 2

    # Values should be finite
    assert np.isfinite(normalized1)
    assert np.isfinite(normalized2)


def test_reward_normalizer_get_stats():
    """Test RewardNormalizer get_stats."""
    normalizer = RewardNormalizer()

    normalizer.update(1.0)
    normalizer.update(2.0)

    stats = normalizer.get_stats()

    assert "mean" in stats
    assert "std" in stats
    assert "count" in stats
    assert stats["count"] == 2


def test_reward_normalizer_single_value():
    """Test RewardNormalizer with single value."""
    normalizer = RewardNormalizer()

    normalizer.update(5.0)

    # Should use default std of 1.0 for single value
    assert normalizer.std == 1.0
    assert normalizer.mean == 5.0


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_reward_with_no_neighbors(mock_adata, gene_sets):
    """Test reward computation without neighbors graph."""
    mock_adata.obs["clusters"] = np.random.choice([0, 1, 2], size=mock_adata.n_obs)
    calculator = RewardCalculator(mock_adata, gene_sets)

    # Should not raise, modularity will be 0
    reward, info = calculator.compute_reward(mock_adata)
    assert np.isfinite(reward)
    assert info["modularity"] == 0.0


def test_reward_with_expression_only():
    """Test reward computation with expression matrix only (no X_scvi)."""
    adata = AnnData(X=np.random.randn(50, 20))
    adata.var_names = [f"gene_{i}" for i in range(20)]
    adata.obs["clusters"] = np.random.choice([0, 1], size=50)

    gene_sets = {"set1": ["gene_0", "gene_1"]}
    calculator = RewardCalculator(adata, gene_sets)

    reward, info = calculator.compute_reward(adata)
    assert np.isfinite(reward)


def test_reward_with_invalid_gene_sets(clustered_adata):
    """Test reward computation with invalid gene sets."""
    gene_sets = {"invalid_set": ["invalid_gene_1", "invalid_gene_2"]}
    calculator = RewardCalculator(clustered_adata, gene_sets)

    reward, info = calculator.compute_reward(clustered_adata)
    assert np.isfinite(reward)
    assert info["Q_GAG"] == 0.0  # No valid genes, no GAG contribution


# ============================================================================
# Test Notebook Alignment
# ============================================================================


def test_reward_formula_matches_notebook_structure(clustered_adata, gene_sets):
    """Test that reward formula matches notebook Cell 23 structure."""
    # Use absolute mode to match original formula
    calculator = RewardCalculator(
        clustered_adata, gene_sets, alpha=0.6, beta=0.4, delta=1.0, reward_mode="absolute"
    )

    reward, info = calculator.compute_reward(clustered_adata)

    # Verify formula: R = α·Q_cluster + β·Q_GAG_transformed - δ·Penalty
    Q_GAG_used = info.get("Q_GAG_transformed", info["Q_GAG"])
    expected = 0.6 * info["Q_cluster"] + 0.4 * Q_GAG_used - 1.0 * info["penalty"]
    assert reward == pytest.approx(expected, rel=1e-6)


def test_q_cluster_weights_match_notebook(clustered_adata, gene_sets):
    """Test Q_cluster weights match notebook: 0.5*silhouette_for_reward + 0.3*modularity + 0.2*balance."""
    calculator = RewardCalculator(clustered_adata, gene_sets)

    Q_cluster, info = calculator._compute_q_cluster(clustered_adata)

    # Notebook formula uses silhouette_for_reward (shifted), not raw silhouette
    expected = (
        0.5 * info["silhouette_for_reward"]
        + 0.3 * info["modularity"]
        + 0.2 * info["balance"]
    )
    assert Q_cluster == pytest.approx(expected, rel=1e-6)


def test_q_gag_normalization_matches_notebook(clustered_adata, gene_sets):
    """Test Q_GAG normalization matches notebook: log1p(f_stat) / 10.0."""
    calculator = RewardCalculator(clustered_adata, gene_sets)

    Q_GAG, info = calculator._compute_q_gag(clustered_adata)

    # Notebook normalization
    f_stats = list(info["f_stats"].values())
    if f_stats:
        f_stats_normalized = [np.log1p(f) / 10.0 for f in f_stats]
        expected = np.mean(f_stats_normalized)
        assert Q_GAG == pytest.approx(expected, rel=1e-6)


def test_penalty_thresholds_match_notebook(mock_adata, gene_sets):
    """Test penalty thresholds match notebook."""
    # Single cluster threshold
    mock_adata.obs["clusters"] = 0
    calculator = RewardCalculator(mock_adata, gene_sets)
    penalty, info = calculator._compute_penalty(mock_adata)
    assert info["n_clusters"] == 1
    # Should have +1.0 for n_clusters == 1

    # Too many clusters threshold (> 0.3 * n_cells)
    mock_adata.obs["clusters"] = np.arange(100) % 40  # 40 clusters
    penalty, info = calculator._compute_penalty(mock_adata)
    assert info["n_clusters"] > 0.3 * 100
    # Should have +1.0 for too many clusters

    # Singleton threshold (< 10 cells)
    labels = np.zeros(100, dtype=int)
    labels[0:5] = [0, 1, 2, 3, 4]
    labels[5:] = 5
    mock_adata.obs["clusters"] = labels
    penalty, info = calculator._compute_penalty(mock_adata)
    assert info["n_singletons"] == 5
    # Should have +0.1 per singleton


# ============================================================================
# Test New Reward Modes
# ============================================================================


def test_reward_mode_absolute(clustered_adata, gene_sets):
    """Test absolute reward mode."""
    calculator = RewardCalculator(
        clustered_adata, gene_sets, reward_mode="absolute", alpha=0.6, beta=0.4, delta=1.0
    )

    reward, info = calculator.compute_reward(clustered_adata)

    # Absolute mode: R = α·Q_cluster + β·Q_GAG_transformed - δ·Penalty
    expected = (
        0.6 * info["Q_cluster"]
        + 0.4 * info.get("Q_GAG_transformed", info["Q_GAG"])
        - 1.0 * info["penalty"]
    )
    assert reward == pytest.approx(expected, rel=1e-6)
    assert info["reward_mode"] == "absolute"


def test_reward_mode_improvement(clustered_adata, gene_sets):
    """Test improvement reward mode."""
    calculator = RewardCalculator(
        clustered_adata,
        gene_sets,
        reward_mode="improvement",
        exploration_bonus=0.2,
        alpha=0.2,
        beta=1.0,
        delta=0.01,
    )

    # First step: should use absolute potential + exploration bonus
    reward1, info1 = calculator.compute_reward(clustered_adata, action=0, current_step=1)
    assert info1["reward_mode"] == "improvement"

    # Second step: should be improvement + exploration bonus
    reward2, info2 = calculator.compute_reward(clustered_adata, action=1, current_step=2)
    assert reward2 >= reward1 - 10.0  # Allow some variance, but should be reasonable


def test_reward_mode_shaped(clustered_adata, gene_sets):
    """Test shaped reward mode (default)."""
    calculator = RewardCalculator(
        clustered_adata, gene_sets, reward_mode="shaped", alpha=0.2, beta=1.0, delta=0.01
    )

    reward, info = calculator.compute_reward(clustered_adata)

    assert info["reward_mode"] == "shaped"
    assert "baseline" in info
    # Shaped mode should keep rewards mostly non-negative after baseline subtraction
    assert reward >= -10.0  # Allow some negative but not extreme


def test_gag_nonlinear_transformation(clustered_adata, gene_sets):
    """Test GAG non-linear transformation."""
    calculator_linear = RewardCalculator(
        clustered_adata, gene_sets, gag_nonlinear=False, gag_scale=6.0
    )
    calculator_nonlinear = RewardCalculator(
        clustered_adata, gene_sets, gag_nonlinear=True, gag_scale=6.0
    )

    reward_linear, info_linear = calculator_linear.compute_reward(clustered_adata)
    reward_nonlinear, info_nonlinear = calculator_nonlinear.compute_reward(clustered_adata)

    # Q_GAG should be same (raw)
    assert info_linear["Q_GAG"] == pytest.approx(info_nonlinear["Q_GAG"], rel=1e-6)

    # Q_GAG_transformed should be different
    if info_nonlinear.get("Q_GAG_transformed") is not None:
        expected_transformed = (info_nonlinear["Q_GAG"] * 6.0) ** 2
        assert info_nonlinear["Q_GAG_transformed"] == pytest.approx(expected_transformed, rel=1e-6)


def test_raw_silhouette_preserved(clustered_adata, gene_sets):
    """Test that raw silhouette is preserved in info dict."""
    calculator = RewardCalculator(clustered_adata, gene_sets, silhouette_shift=0.5)

    reward, info = calculator.compute_reward(clustered_adata)

    # Raw silhouette should be in info
    assert "silhouette" in info
    assert "silhouette_for_reward" in info

    # Raw silhouette should be original value (can be negative)
    raw_silhouette = info["silhouette"]
    assert -1.0 <= raw_silhouette <= 1.0

    # Silhouette for reward should be shifted
    silhouette_for_reward = info["silhouette_for_reward"]
    assert silhouette_for_reward >= 0.0  # Should be non-negative after shift


def test_reward_calculator_reset(clustered_adata, gene_sets):
    """Test reward calculator reset."""
    calculator = RewardCalculator(
        clustered_adata, gene_sets, reward_mode="improvement", exploration_bonus=0.2
    )

    # Compute reward to set internal state
    calculator.compute_reward(clustered_adata, action=0, current_step=1)
    assert calculator._previous_potential is not None

    # Reset should clear internal state
    calculator.reset()
    assert calculator._previous_potential is None


def test_early_termination_penalty(clustered_adata, gene_sets):
    """Test early termination penalty flag."""
    calculator = RewardCalculator(
        clustered_adata,
        gene_sets,
        early_termination_penalty=-5.0,
        min_steps_before_accept=20,
    )

    # Accept action before minimum steps should flag penalty
    reward_early, info_early = calculator.compute_reward(
        clustered_adata, action=4, current_step=10
    )
    assert info_early["early_termination_penalty_applied"] is True
    # Reward should be the computed reward (penalty applied in ClusteringEnv)
    assert reward_early != -5.0  # Penalty not applied here, only flagged

    # Accept action after minimum steps should not flag penalty
    reward_late, info_late = calculator.compute_reward(
        clustered_adata, action=4, current_step=25
    )
    assert info_late["early_termination_penalty_applied"] is False
    assert reward_late != -5.0

    # Non-accept action should not flag penalty
    reward_normal, info_normal = calculator.compute_reward(
        clustered_adata, action=0, current_step=10
    )
    assert info_normal["early_termination_penalty_applied"] is False
    assert reward_normal != -5.0


def test_reward_calculator_new_parameters(clustered_adata, gene_sets):
    """Test RewardCalculator with all new parameters."""
    calculator = RewardCalculator(
        clustered_adata,
        gene_sets,
        reward_mode="shaped",
        gag_nonlinear=True,
        gag_scale=6.0,
        exploration_bonus=0.2,
        silhouette_shift=0.5,
        early_termination_penalty=-5.0,
        min_steps_before_accept=20,
    )

    assert calculator.reward_mode == "shaped"
    assert calculator.gag_nonlinear == True
    assert calculator.gag_scale == 6.0
    assert calculator.exploration_bonus == 0.2
    assert calculator.silhouette_shift == 0.5
    assert calculator.early_termination_penalty == -5.0
    assert calculator.min_steps_before_accept == 20

    # Should compute reward successfully
    reward, info = calculator.compute_reward(clustered_adata, action=0, current_step=5)
    assert np.isfinite(reward)
    assert info["reward_mode"] == "shaped"
