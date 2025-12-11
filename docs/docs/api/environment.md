# Environment API Reference

## ClusteringEnv

::: rl_sc_cluster_utils.environment.ClusteringEnv
    options:
      show_source: true
      heading_level: 3

## RewardCalculator

::: rl_sc_cluster_utils.environment.RewardCalculator
    options:
      show_source: true
      heading_level: 3

## RewardNormalizer

::: rl_sc_cluster_utils.environment.RewardNormalizer
    options:
      show_source: true
      heading_level: 3

## ActionExecutor

::: rl_sc_cluster_utils.environment.ActionExecutor
    options:
      show_source: true
      heading_level: 3

## Overview

The `ClusteringEnv` class is a Gymnasium-compatible reinforcement learning environment for scRNA-seq cluster refinement.

### Key Features

- **Gymnasium Compliance**: Fully compatible with the Gymnasium API
- **35-Dimensional State Space**: Encodes clustering state with global metrics, quality metrics, and GAG enrichment
- **5 Discrete Actions**: Split, merge, re-cluster (up/down), and accept
- **Composite Reward**: Q_cluster + Q_GAG - Penalty (Stage 4 complete)
- **Episodic Learning**: Episodes terminate on Accept action or max steps
- **Configurable**: Optional state/reward normalization

### Quick Example

```python
from anndata import AnnData
import numpy as np
from rl_sc_cluster_utils.environment import ClusteringEnv

# Create mock data
adata = AnnData(X=np.random.randn(100, 50))
adata.var_names = [f"gene_{i}" for i in range(50)]

# Define gene sets
gene_sets = {
    "set1": ["gene_0", "gene_1", "gene_2"],
    "set2": ["gene_10", "gene_11", "gene_12"],
}

# Initialize environment
env = ClusteringEnv(
    adata=adata,
    gene_sets=gene_sets,
    max_steps=15,
    normalize_state=False,
    normalize_rewards=True  # Default: enabled for stable training
)

# Use the environment
state, info = env.reset()
state, reward, terminated, truncated, info = env.step(0)

# Access reward components
print(f"Reward: {reward:.4f}")
print(f"Q_cluster: {info['Q_cluster']:.4f}")
print(f"Q_GAG: {info['Q_GAG']:.4f}")
print(f"Penalty: {info['penalty']:.4f}")
```

## State Space

The observation space is a 35-dimensional continuous vector:

| Dimensions | Component | Description |
|------------|-----------|-------------|
| 0-2 | Global Metrics | Cluster count, mean size, entropy |
| 3-5 | Quality Metrics | Silhouette, modularity, balance |
| 6-33 | GAG Enrichment | 7 gene sets × 4 metrics each |
| 34 | Progress | Normalized episode progress |

See [State Representation](../environment/state_representation.md) for details.

## Action Space

The action space is discrete with 5 actions:

| Action | Name | Description |
|--------|------|-------------|
| 0 | Split | Split worst cluster by silhouette |
| 1 | Merge | Merge closest pair by centroid distance |
| 2 | Re-cluster+ | Increase resolution by 0.1 |
| 3 | Re-cluster- | Decrease resolution by 0.1 |
| 4 | Accept | Terminate episode with current clustering |

See [Action Implementation](../environment/action_implementation.md) for details.

## Reward Function

The reward is a composite of:

```
R = α·Q_cluster + β·Q_GAG - δ·Penalty
```

Where:
- `Q_cluster = 0.5·silhouette + 0.3·modularity + 0.2·balance`
- `Q_GAG = mean(log1p(f_stat) / 10.0)` across gene sets
- `Penalty = degenerate_penalty + singleton_penalty + bounds_penalty`

Default weights: α=0.2, β=2.0, δ=0.01 (GAG-focused configuration)
Default reward mode: "shaped" (avoids negative rewards)

See [Reward Calculation](../environment/reward_calculation.md) for details.

## Methods

### `__init__(adata, gene_sets=None, max_steps=15, normalize_state=False, normalize_rewards=True, render_mode=None, reward_alpha=0.2, reward_beta=2.0, reward_delta=0.01, reward_mode="shaped", gag_nonlinear=True, gag_scale=6.0, exploration_bonus=0.2, silhouette_shift=0.5, early_termination_penalty=-5.0, min_steps_before_accept=20)`

Initialize the environment.

**Parameters:**

- `adata` (AnnData): Annotated data object with single-cell data
- `gene_sets` (dict, optional): Dictionary mapping gene set names to gene lists
- `max_steps` (int): Maximum steps per episode (default: 15)
- `normalize_state` (bool): Whether to normalize state vector (default: False)
- `normalize_rewards` (bool): Whether to normalize rewards (default: True)
- `render_mode` (str, optional): Render mode for visualization
- `reward_alpha` (float): Weight for Q_cluster (default: 0.2)
- `reward_beta` (float): Weight for Q_GAG (default: 2.0)
- `reward_delta` (float): Weight for penalty (default: 0.01)
- `reward_mode` (str): Reward mode: "absolute", "improvement", or "shaped" (default: "shaped")
- `gag_nonlinear` (bool): Apply non-linear GAG transformation (default: True)
- `gag_scale` (float): Scaling factor for GAG transformation (default: 6.0)
- `exploration_bonus` (float): Bonus per step for improvement mode (default: 0.2)
- `silhouette_shift` (float): Shift amount to keep silhouette non-negative (default: 0.5)
- `early_termination_penalty` (float): Penalty for Accept action before minimum steps (default: -5.0)
- `min_steps_before_accept` (int): Minimum steps before Accept action allowed without penalty (default: 20)

**Returns:** ClusteringEnv instance

### `reset(seed=None, options=None)`

Reset the environment to initial state.

**Parameters:**

- `seed` (int, optional): Random seed for reproducibility
- `options` (dict, optional): Additional reset options

**Returns:**

- `state` (np.ndarray): Initial state vector (35 dimensions)
- `info` (dict): Additional information

### `step(action)`

Execute one step in the environment.

**Parameters:**

- `action` (int): Action to take (0-4)

**Returns:**

- `state` (np.ndarray): Next state vector (35 dimensions)
- `reward` (float): Composite reward (may be normalized if enabled)
- `terminated` (bool): Whether episode is terminated (Accept action)
- `truncated` (bool): Whether episode is truncated (max steps)
- `info` (dict): Additional information including:
  - `action_success` (bool): Whether action executed successfully
  - `action_error` (str or None): Error message if action failed
  - `resolution_clamped` (bool): Whether resolution was clamped to bounds
  - `no_change` (bool): Whether action had no effect
  - `n_clusters` (int): Number of clusters after action
  - `resolution` (float): Current resolution after action
  - `raw_reward` (float): Reward before normalization
  - `Q_cluster` (float): Clustering quality score
  - `Q_GAG` (float): GAG enrichment score
  - `penalty` (float): Total penalty
  - `silhouette` (float): Silhouette score
  - `modularity` (float): Graph modularity
  - `balance` (float): Cluster balance
  - `n_singletons` (int): Number of singleton clusters
  - `mean_f_stat` (float): Mean F-statistic across gene sets

### `render()`

Render the environment (stub for future visualization).

**Returns:** None (visualization not yet implemented)

### `close()`

Clean up resources.

## Attributes

### `action_space`

Discrete action space with 5 actions.

**Type:** `gymnasium.spaces.Discrete(5)`

### `observation_space`

Continuous observation space with 35 dimensions.

**Type:** `gymnasium.spaces.Box(low=-inf, high=inf, shape=(35,), dtype=float64)`

### `max_steps`

Maximum number of steps per episode.

**Type:** int

### `current_step`

Current step in the episode.

**Type:** int

### `current_resolution`

Current Leiden clustering resolution.

**Type:** float

### `reward_calculator`

RewardCalculator instance for computing rewards.

**Type:** RewardCalculator

### `reward_normalizer`

RewardNormalizer instance (if `normalize_rewards=True`).

**Type:** Optional[RewardNormalizer]

## Usage Examples

### Basic Episode

```python
env = ClusteringEnv(adata, gene_sets=gene_sets)
state, info = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward}")
print(f"Final Q_cluster: {info['Q_cluster']:.4f}")
```

### With Reward Normalization

```python
env = ClusteringEnv(
    adata,
    gene_sets=gene_sets,
    normalize_rewards=True  # Enable normalization
)
state, info = env.reset()

for _ in range(10):
    state, reward, terminated, truncated, info = env.step(2)
    print(f"Normalized reward: {reward:.4f}, Raw: {info['raw_reward']:.4f}")
    if terminated or truncated:
        break
```

### Custom Reward Weights

```python
env = ClusteringEnv(
    adata,
    gene_sets=gene_sets,
    reward_alpha=0.5,   # Clustering quality weight
    reward_beta=0.3,    # GAG enrichment weight
    reward_delta=0.8    # Penalty weight
)
```

### With PPO Training

```python
from stable_baselines3 import PPO

env = ClusteringEnv(adata, gene_sets=gene_sets)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
```

### Multiple Episodes

```python
env = ClusteringEnv(adata, gene_sets=gene_sets, max_steps=15)

for episode in range(10):
    state, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    print(f"Episode {episode}: reward={episode_reward:.4f}")
```

## Current Implementation Status

**Stage 4 Complete**:
- ✅ **State**: Real 35-dimensional state vector computed from clustering metrics
- ✅ **Actions**: All 5 actions fully implemented and functional
  - Action 0: Split worst cluster (by silhouette)
  - Action 1: Merge closest pair (by centroid distance)
  - Action 2: Re-cluster resolution +0.1 (with clamping)
  - Action 3: Re-cluster resolution -0.1 (with clamping)
  - Action 4: Accept (terminate episode)
- ✅ **Reward**: Composite reward function implemented
  - Q_cluster: Silhouette + modularity + balance
  - Q_GAG: Normalized F-statistics across gene sets
  - Penalty: Degenerate states, singletons, bounds
- ✅ **Normalization**: Optional reward normalization with running statistics

**Next Steps**:
- Stage 5: Training and evaluation

## See Also

- [Development Plan](../environment/development_plan.md) - Full implementation roadmap
- [Design Decisions](../environment/design_decisions.md) - Rationale for key choices
- [Reward Calculation](../environment/reward_calculation.md) - Detailed reward documentation
- [Testing Guide](../dev/testing.md) - How to test the environment
