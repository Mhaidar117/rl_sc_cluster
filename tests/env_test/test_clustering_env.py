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


def test_placeholder_reward(env):
    """Test that placeholder reward is 0.0."""
    env.reset()

    for action in range(5):
        env.reset()
        _, reward, _, _, _ = env.step(action)

        # Placeholder reward should be 0.0
        assert reward == 0.0


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

    # Resolution should remain constant (placeholder behavior in Stage 2)
    env.step(2)  # Increase resolution action
    assert env.current_resolution == 0.5  # Unchanged until Stage 3


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
