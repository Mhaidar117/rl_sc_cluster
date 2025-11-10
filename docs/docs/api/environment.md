# Environment API Reference

## ClusteringEnv

::: rl_sc_cluster_utils.environment.ClusteringEnv
    options:
      show_source: true
      heading_level: 3

## Overview

The `ClusteringEnv` class is a Gymnasium-compatible reinforcement learning environment for scRNA-seq cluster refinement.

### Key Features

- **Gymnasium Compliance**: Fully compatible with the Gymnasium API
- **35-Dimensional State Space**: Encodes clustering state with global metrics, quality metrics, and GAG enrichment
- **5 Discrete Actions**: Split, merge, re-cluster (up/down), and accept
- **Episodic Learning**: Episodes terminate on Accept action or max steps
- **Configurable**: Optional state/reward normalization

### Quick Example

```python
from anndata import AnnData
import numpy as np
from rl_sc_cluster_utils.environment import ClusteringEnv

# Create mock data
adata = AnnData(X=np.random.randn(100, 50))

# Initialize environment
env = ClusteringEnv(
    adata=adata,
    max_steps=15,
    normalize_state=False,
    normalize_rewards=False
)

# Use the environment
state, info = env.reset()
state, reward, terminated, truncated, info = env.step(0)
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
- `Q_cluster`: Clustering quality (silhouette, modularity, balance)
- `Q_GAG`: GAG enrichment separation (ANOVA F-stat, mutual info)
- `Penalty`: Degenerate states, quality degradation, bounds violations

Default weights: α=0.6, β=0.4, δ=1.0

See [Reward Calculation](../environment/reward_calculation.md) for details.

## Methods

### `__init__(adata, max_steps=15, normalize_state=False, normalize_rewards=False, render_mode=None)`

Initialize the environment.

**Parameters:**

- `adata` (AnnData): Annotated data object with single-cell data
- `max_steps` (int): Maximum steps per episode (default: 15)
- `normalize_state` (bool): Whether to normalize state vector (default: False)
- `normalize_rewards` (bool): Whether to normalize rewards (default: False)
- `render_mode` (str, optional): Render mode for visualization

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
- `reward` (float): Reward for the action
- `terminated` (bool): Whether episode is terminated (Accept action)
- `truncated` (bool): Whether episode is truncated (max steps)
- `info` (dict): Additional information

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

## Usage Examples

### Basic Episode

```python
env = ClusteringEnv(adata)
state, info = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward}")
```

### With PPO Training

```python
from stable_baselines3 import PPO

env = ClusteringEnv(adata)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
```

### Multiple Episodes

```python
env = ClusteringEnv(adata, max_steps=15)

for episode in range(10):
    state, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    print(f"Episode {episode}: reward={episode_reward}")
```

## Current Implementation Status

**Stage 2 Complete**:
- ✅ **State**: Real 35-dimensional state vector computed from clustering metrics
- 🔲 **Actions**: No-op placeholders (will modify clustering in Stage 3)
- 🔲 **Reward**: Returns 0.0 placeholder (will compute composite reward in Stage 4)

**Next Steps**:
- Stage 3: Implement actions (split, merge, re-cluster)
- Stage 4: Implement composite reward function

## See Also

- [Development Plan](../environment/development_plan.md) - Full implementation roadmap
- [Design Decisions](../environment/design_decisions.md) - Rationale for key choices
- [Testing Guide](../dev/testing.md) - How to test the environment
