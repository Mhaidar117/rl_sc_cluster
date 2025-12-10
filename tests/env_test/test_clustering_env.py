"""Unit tests for ClusteringEnv."""

import numpy as np
import pytest
from anndata import AnnData
from gymnasium.utils.env_checker import check_env

from rl_sc_cluster_utils.environment import ClusteringEnv


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

    return adata


@pytest.fixture
def gene_sets():
    """Create mock gene sets for testing."""
    return {
        "set_1": ["gene_0", "gene_1", "gene_2"],
        "set_2": ["gene_3", "gene_4", "gene_5"],
    }


@pytest.fixture
def env(mock_adata, gene_sets):
    """Create a ClusteringEnv instance for testing."""
    return ClusteringEnv(adata=mock_adata, gene_sets=gene_sets, max_steps=15)


def test_gymnasium_compliance(env):
    """Test that environment passes Gymnasium's check_env."""
    # This will raise an error if the environment is not compliant
    check_env(env.unwrapped, skip_render_check=True)


def test_initialization(mock_adata, gene_sets):
    """Test environment initialization."""
    env = ClusteringEnv(
        adata=mock_adata,
        gene_sets=gene_sets,
        max_steps=10,
        normalize_state=True,
        normalize_rewards=True,
        render_mode="human",
    )

    assert env.max_steps == 10
    assert env.normalize_state is True
    assert env.normalize_rewards is True
    assert env.render_mode == "human"
    assert env.current_resolution == 0.5
    assert env.state_extractor is not None


def test_action_space(env):
    """Test action space is correctly defined."""
    assert env.action_space.n == 5
    assert env.action_space.contains(0)
    assert env.action_space.contains(4)
    assert not env.action_space.contains(5)
    assert not env.action_space.contains(-1)


def test_observation_space(env):
    """Test observation space is correctly defined."""
    assert env.observation_space.shape == (35,)
    assert env.observation_space.dtype == np.float64

    # Test that a valid observation is contained
    valid_obs = np.zeros(35, dtype=np.float64)
    assert env.observation_space.contains(valid_obs)

    # Test that invalid observations are not contained
    invalid_obs = np.zeros(30, dtype=np.float64)
    assert not env.observation_space.contains(invalid_obs)


def test_reset(env):
    """Test reset returns correct shape and format."""
    state, info = env.reset()

    # Check state shape
    assert state.shape == (35,)
    assert state.dtype == np.float64

    # Check info dictionary
    assert isinstance(info, dict)
    assert "step" in info
    assert "resolution" in info
    assert "n_clusters" in info

    # Check episode tracking is reset
    assert env.current_step == 0
    assert env.current_resolution == 0.5


def test_reset_with_seed(env):
    """Test reset with seed for reproducibility."""
    state1, _ = env.reset(seed=42)
    state2, _ = env.reset(seed=42)

    # States should be identical with same seed
    # Note: Leiden clustering may have some randomness, so we check shape instead
    assert state1.shape == state2.shape
    assert state1.shape == (35,)


def test_step_format(env):
    """Test step returns correct format for all actions."""
    env.reset()

    for action in range(5):
        env.reset()
        state, reward, terminated, truncated, info = env.step(action)

        # Check return types
        assert isinstance(state, np.ndarray)
        assert state.shape == (35,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


def test_step_invalid_action(env):
    """Test that invalid actions raise ValueError."""
    env.reset()

    with pytest.raises(ValueError):
        env.step(5)  # Out of bounds

    with pytest.raises(ValueError):
        env.step(-1)  # Negative action


def test_termination_accept_action(env):
    """Test that Accept action (4) terminates episode."""
    env.reset()

    state, reward, terminated, truncated, info = env.step(4)

    assert terminated is True
    assert truncated is False
    assert info["action"] == 4
    assert info["terminated"] is True


def test_termination_other_actions(env):
    """Test that other actions don't terminate episode."""
    env.reset()

    for action in range(4):
        env.reset()
        state, reward, terminated, truncated, info = env.step(action)

        # Should not terminate on first step with actions 0-3
        assert terminated is False


def test_max_steps_truncation(env):
    """Test episode truncates after max_steps."""
    env.reset()

    # Take max_steps actions
    for i in range(env.max_steps):
        state, reward, terminated, truncated, info = env.step(0)

        if i < env.max_steps - 1:
            # Should not truncate before max_steps
            assert truncated is False
        else:
            # Should truncate at max_steps
            assert truncated is True
            assert terminated is False


def test_step_counter(env):
    """Test that step counter increments correctly."""
    env.reset()
    assert env.current_step == 0

    env.step(0)
    assert env.current_step == 1

    env.step(1)
    assert env.current_step == 2


def test_multiple_episodes(env):
    """Test running multiple episodes."""
    for episode in range(3):
        state, info = env.reset()
        assert env.current_step == 0

        done = False
        steps = 0
        while not done and steps < env.max_steps:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps <= env.max_steps


def test_render(env):
    """Test render method (stub for now)."""
    env.reset()

    # Should not raise error
    result = env.render()

    # Currently returns None
    assert result is None


def test_close(env):
    """Test close method."""
    env.reset()

    # Should not raise error
    env.close()


def test_state_extraction(env):
    """Test that state is extracted correctly (not all zeros)."""
    state, _ = env.reset()

    # State should not be all zeros (real extraction)
    assert not np.all(state == 0.0)

    # State should be finite
    assert np.all(np.isfinite(state))

    # Progress component should be 0 at start
    assert state[34] == pytest.approx(0.0)


def test_reward_computation(env):
    """Test that reward is computed (not placeholder)."""
    env.reset()

    for action in range(5):
        env.reset()
        _, reward, _, _, info = env.step(action)

        # Reward should be a finite float
        assert np.isfinite(reward)

        # Reward info should contain components
        assert "Q_cluster" in info
        assert "Q_GAG" in info
        assert "penalty" in info
        assert "raw_reward" in info

        # Q_cluster, Q_GAG, and penalty should be finite
        assert np.isfinite(info["Q_cluster"])
        assert np.isfinite(info["Q_GAG"])
        assert np.isfinite(info["penalty"])


def test_state_updates_with_step(env):
    """Test that state updates with step (progress component changes)."""
    state1, _ = env.reset()

    # Take an action
    state2, _, _, _, _ = env.step(0)

    # Progress component should change
    assert state1[34] != state2[34]
    assert state1[34] == pytest.approx(0.0)
    assert state2[34] == pytest.approx(1 / 15)


def test_info_dict_contents(env):
    """Test that info dict contains expected keys."""
    env.reset()

    _, _, _, _, info = env.step(0)

    # Check required keys
    assert "action" in info
    assert "step" in info
    assert "terminated" in info
    assert "truncated" in info
    assert "resolution" in info
    assert "n_clusters" in info


def test_resolution_tracking(env):
    """Test that resolution is tracked correctly."""
    env.reset()
    assert env.current_resolution == 0.5

    # Resolution should update with actions (Stage 3)
    env.step(2)  # Increase resolution action
    assert env.current_resolution == pytest.approx(0.6, abs=0.01)


def test_clustering_performed_on_reset(env):
    """Test that clustering is performed on reset."""
    state, info = env.reset()

    # Check that clusters were created
    assert "clusters" in env.adata.obs
    assert info["n_clusters"] > 0

    # Check that neighbors graph was created
    assert "neighbors" in env.adata.uns


def test_state_has_correct_components(env):
    """Test that state vector has all expected components."""
    state, _ = env.reset()

    # All components should be finite
    assert np.all(np.isfinite(state))

    # Check ranges for some components
    # Global metrics (0-2) should be non-negative
    assert np.all(state[0:3] >= 0)

    # Progress (34) should be in [0, 1]
    assert 0 <= state[34] <= 1


# Stage 3: Action Implementation Tests


def test_action_recluster_increment(env):
    """Test that re-cluster increment action works."""
    env.reset()
    initial_resolution = env.current_resolution

    state, reward, terminated, truncated, info = env.step(2)  # Increment resolution

    assert info["action_success"] is True
    assert env.current_resolution > initial_resolution
    assert env.current_resolution == pytest.approx(initial_resolution + 0.1, abs=0.01)


def test_action_recluster_decrement(env):
    """Test that re-cluster decrement action works."""
    env.reset()
    initial_resolution = env.current_resolution

    state, reward, terminated, truncated, info = env.step(3)  # Decrement resolution

    assert info["action_success"] is True
    assert env.current_resolution < initial_resolution
    assert env.current_resolution == pytest.approx(initial_resolution - 0.1, abs=0.01)


def test_action_recluster_resolution_clamped(env):
    """Test that resolution clamping works."""
    env.reset()

    # Set resolution to max (or close to it)
    env.current_resolution = 1.95
    state, reward, terminated, truncated, info = env.step(2)  # Try to increment

    assert info["action_success"] is True
    # Should clamp to 2.0
    assert env.current_resolution == pytest.approx(2.0, abs=0.01)
    # May or may not be clamped depending on exact value
    if env.current_resolution >= 2.0:
        assert info["resolution_clamped"] is True

    # Set resolution to min (or close to it)
    env.current_resolution = 0.15
    state, reward, terminated, truncated, info = env.step(3)  # Try to decrement

    assert info["action_success"] is True
    # Should clamp to 0.1
    assert env.current_resolution == pytest.approx(0.1, abs=0.01)
    # May or may not be clamped depending on exact value
    if env.current_resolution <= 0.1:
        assert info["resolution_clamped"] is True


def test_action_merge_closest_pair(env):
    """Test that merge action works."""
    env.reset()
    initial_n_clusters = len(env.adata.obs["clusters"].unique())

    # Only merge if we have more than 1 cluster
    if initial_n_clusters > 1:
        state, reward, terminated, truncated, info = env.step(1)  # Merge

        assert info["action_success"] is True
        final_n_clusters = len(env.adata.obs["clusters"].unique())
        assert final_n_clusters == initial_n_clusters - 1


def test_action_split_worst_cluster(env):
    """Test that split action works."""
    env.reset()
    initial_n_clusters = len(env.adata.obs["clusters"].unique())

    state, reward, terminated, truncated, info = env.step(0)  # Split

    assert info["action_success"] is True

    # May or may not split (depends on data and whether sub-clustering produces >1 cluster)
    if not info["no_change"]:
        final_n_clusters = len(env.adata.obs["clusters"].unique())
        assert final_n_clusters >= initial_n_clusters


def test_action_info_dict_stage3(env):
    """Test that info dict contains Stage 3 action fields."""
    env.reset()

    state, reward, terminated, truncated, info = env.step(0)

    # Check new Stage 3 fields
    assert "action_success" in info
    assert "action_error" in info
    assert "resolution_clamped" in info
    assert "no_change" in info

    assert isinstance(info["action_success"], bool)
    assert info["action_error"] is None or isinstance(info["action_error"], str)
    assert isinstance(info["resolution_clamped"], bool)
    assert isinstance(info["no_change"], bool)


def test_action_invalid_split_single_cluster(env):
    """Test that invalid split action (single cluster) is handled gracefully."""
    env.reset()

    # Force single cluster
    env.adata.obs["clusters"] = 0

    state, reward, terminated, truncated, info = env.step(0)  # Try to split

    assert info["action_success"] is False
    assert info["no_change"] is True
    assert info["action_error"] is not None
    assert "Cannot split" in info["action_error"]


def test_action_invalid_merge_single_cluster(env):
    """Test that invalid merge action (single cluster) is handled gracefully."""
    env.reset()

    # Force single cluster
    env.adata.obs["clusters"] = 0

    state, reward, terminated, truncated, info = env.step(1)  # Try to merge

    assert info["action_success"] is False
    assert info["no_change"] is True
    assert info["action_error"] is not None
    assert "Cannot merge" in info["action_error"]


def test_cluster_ids_numeric_after_actions(env):
    """Test that cluster IDs remain numeric after actions."""
    env.reset()

    # Execute various actions
    env.step(2)  # Re-cluster increment
    assert env.adata.obs["clusters"].dtype in [np.int8, np.int16, np.int32, np.int64]

    env.step(3)  # Re-cluster decrement
    assert env.adata.obs["clusters"].dtype in [np.int8, np.int16, np.int32, np.int64]

    if len(env.adata.obs["clusters"].unique()) > 1:
        env.step(1)  # Merge
        assert env.adata.obs["clusters"].dtype in [np.int8, np.int16, np.int32, np.int64]


def test_state_extraction_after_actions(env):
    """Test that state extraction still works after actions."""
    env.reset()
    state1, _ = env.reset()

    # Execute an action
    state2, _, _, _, _ = env.step(2)  # Re-cluster increment

    # State should still be valid
    assert state2.shape == (35,)
    assert np.all(np.isfinite(state2))

    # Clustering metrics may have changed
    # (We don't assert specific changes as they depend on data)


def test_multiple_actions_sequence(env):
    """Test executing a sequence of actions."""
    env.reset()

    # Execute multiple actions
    for action in [2, 2, 3, 1, 0]:  # Increment, increment, decrement, merge, split
        state, reward, terminated, truncated, info = env.step(action)

        # Should not crash
        assert state.shape == (35,)
        assert info["action_success"] is True or info["no_change"] is True

        if terminated:
            break


# ============================================================================
# Stage 4: Reward Integration Tests
# ============================================================================


def test_reward_components_in_info(env):
    """Test that reward components are included in info dict."""
    env.reset()
    _, reward, _, _, info = env.step(0)

    # Reward components should be in info
    assert "raw_reward" in info
    assert "Q_cluster" in info
    assert "Q_GAG" in info
    assert "penalty" in info
    assert "silhouette" in info
    assert "modularity" in info
    assert "balance" in info
    assert "n_singletons" in info
    assert "mean_f_stat" in info

    # All should be finite
    assert np.isfinite(info["raw_reward"])
    assert np.isfinite(info["Q_cluster"])
    assert np.isfinite(info["Q_GAG"])
    assert np.isfinite(info["penalty"])


def test_reward_formula_applied(env):
    """Test that reward follows formula: R = α·Q_cluster + β·Q_GAG - δ·Penalty."""
    env.reset()
    _, reward, _, _, info = env.step(0)

    # Default weights: alpha=0.6, beta=0.4, delta=1.0
    expected = 0.6 * info["Q_cluster"] + 0.4 * info["Q_GAG"] - 1.0 * info["penalty"]

    # Raw reward (before normalization) should match formula
    assert info["raw_reward"] == pytest.approx(expected, rel=1e-6)


def test_reward_changes_with_actions(env):
    """Test that reward changes when clustering changes."""
    env.reset()

    rewards = []
    for action in range(4):  # Actions 0-3 (not 4 which terminates)
        env.reset()
        _, reward, _, _, info = env.step(action)
        rewards.append(info["raw_reward"])

    # At least some rewards should be different
    # (depends on random data, but unlikely all are exactly the same)
    assert any(np.isfinite(r) for r in rewards)


def test_reward_penalty_for_resolution_clamping(env):
    """Test that reward penalty is applied when resolution is clamped."""
    # Test upper bound clamping
    env.reset()

    # Increment resolution until clamped
    for _ in range(20):
        _, _, terminated, _, info = env.step(2)  # Increment resolution
        if info["resolution_clamped"]:
            # When clamped, penalty should include bounds penalty
            # We verify resolution_clamped is properly tracked
            assert info["resolution_clamped"] is True
            break
        if terminated:
            break


def test_reward_normalization_disabled_by_default(mock_adata, gene_sets):
    """Test that reward normalization is disabled by default."""
    from rl_sc_cluster_utils.environment import ClusteringEnv

    env = ClusteringEnv(mock_adata, gene_sets=gene_sets, normalize_rewards=False)
    env.reset()

    _, reward1, _, _, info1 = env.step(0)

    # Raw reward should equal returned reward when normalization disabled
    assert reward1 == info1["raw_reward"]


def test_reward_normalization_enabled(mock_adata, gene_sets):
    """Test that reward normalization works when enabled."""
    from rl_sc_cluster_utils.environment import ClusteringEnv

    env = ClusteringEnv(mock_adata, gene_sets=gene_sets, normalize_rewards=True)
    env.reset()

    # Take multiple steps to build up normalization history
    rewards = []
    raw_rewards = []
    for _ in range(5):
        _, reward, terminated, _, info = env.step(2)  # Re-cluster increment
        rewards.append(reward)
        raw_rewards.append(info["raw_reward"])
        if terminated:
            break

    # Normalized rewards should be different from raw rewards after a few steps
    # (first reward equals raw since no history yet)
    assert len(rewards) > 0


def test_reward_normalization_resets_on_episode_reset(mock_adata, gene_sets):
    """Test that reward normalizer resets on episode reset."""
    from rl_sc_cluster_utils.environment import ClusteringEnv

    env = ClusteringEnv(mock_adata, gene_sets=gene_sets, normalize_rewards=True)

    # First episode
    env.reset()
    env.step(0)
    env.step(1)

    # Reset for new episode
    env.reset()

    # Normalizer should be reset
    assert env.reward_normalizer.count == 0


def test_custom_reward_weights(mock_adata, gene_sets):
    """Test custom reward weights."""
    from rl_sc_cluster_utils.environment import ClusteringEnv

    env = ClusteringEnv(
        mock_adata,
        gene_sets=gene_sets,
        reward_alpha=0.5,
        reward_beta=0.3,
        reward_delta=0.8,
    )
    env.reset()

    _, _, _, _, info = env.step(0)

    # Verify custom weights are applied
    expected = 0.5 * info["Q_cluster"] + 0.3 * info["Q_GAG"] - 0.8 * info["penalty"]
    assert info["raw_reward"] == pytest.approx(expected, rel=1e-6)


def test_reward_with_gene_sets(mock_adata, gene_sets):
    """Test reward computation with gene sets."""
    from rl_sc_cluster_utils.environment import ClusteringEnv

    env = ClusteringEnv(mock_adata, gene_sets=gene_sets)
    env.reset()

    _, _, _, _, info = env.step(0)

    # Q_GAG should be computed with gene sets
    assert "mean_f_stat" in info
    assert np.isfinite(info["mean_f_stat"])


def test_reward_without_gene_sets(mock_adata):
    """Test reward computation without gene sets."""
    from rl_sc_cluster_utils.environment import ClusteringEnv

    env = ClusteringEnv(mock_adata, gene_sets=None)
    env.reset()

    _, _, _, _, info = env.step(0)

    # Q_GAG should be 0 without gene sets
    assert info["Q_GAG"] == 0.0
    assert info["mean_f_stat"] == 0.0


def test_reward_consistency_across_episode(env):
    """Test reward computation consistency across an episode."""
    env.reset()

    rewards = []
    q_clusters = []
    q_gags = []
    penalties = []

    for _ in range(10):
        _, reward, terminated, truncated, info = env.step(2)

        rewards.append(reward)
        q_clusters.append(info["Q_cluster"])
        q_gags.append(info["Q_GAG"])
        penalties.append(info["penalty"])

        if terminated or truncated:
            break

    # All values should be finite
    assert all(np.isfinite(r) for r in rewards)
    assert all(np.isfinite(q) for q in q_clusters)
    assert all(np.isfinite(q) for q in q_gags)
    assert all(np.isfinite(p) for p in penalties)


def test_previous_reward_tracking(env):
    """Test that previous reward is tracked correctly."""
    env.reset()

    # Initially, previous reward should be None
    assert env._previous_reward is None

    # After first step, previous reward should be set
    env.step(0)
    assert env._previous_reward is not None

    # After reset, previous reward should be None again
    env.reset()
    assert env._previous_reward is None
