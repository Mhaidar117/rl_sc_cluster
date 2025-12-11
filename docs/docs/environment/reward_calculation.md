# Reward Calculation

This document details the composite reward function used in the clustering environment.

**Implementation Status**: Complete (Stage 4)

---

## Overview

The reward function balances three objectives:
1. **Clustering Quality** (Q_cluster): How well-separated and balanced clusters are
2. **GAG Enrichment** (Q_GAG): How well clusters separate by GAG-sulfation expression
3. **Penalties**: Discourage degenerate states and bad actions

**Reward Modes**: The system supports three reward modes:
- **"shaped"** (default): Keeps rewards non-negative by subtracting a running baseline
- **"improvement"**: Delta rewards based on improvement in potential
- **"absolute"**: Direct composite reward without shaping

**Base Formula**:
```python
raw_reward = α * Q_cluster + β * Q_GAG_transformed - δ * Penalty
```

**Default weights**: α = 0.2, β = 2.0, δ = 0.01 (GAG-focused configuration)

---

## Implementation

The reward system is implemented in `rl_sc_cluster_utils/environment/rewards.py` with two main classes:

### RewardCalculator

```python
from rl_sc_cluster_utils.environment import RewardCalculator

calculator = RewardCalculator(
    adata,
    gene_sets,
    alpha=0.2,           # Weight for clustering quality
    beta=2.0,            # Weight for GAG enrichment (high for biology focus)
    delta=0.01,          # Weight for penalties
    reward_mode="shaped",  # Reward mode: "absolute", "improvement", or "shaped"
    gag_nonlinear=True,    # Apply non-linear GAG transformation
    gag_scale=6.0,        # Scaling factor for GAG transformation
    exploration_bonus=0.2, # Bonus for improvement mode
    silhouette_shift=0.5,  # Shift to keep silhouette non-negative
    early_termination_penalty=-5.0,  # Penalty for early Accept
    min_steps_before_accept=20,      # Minimum steps before Accept allowed
)

# Compute reward
reward, info = calculator.compute_reward(
    adata,
    previous_reward=None,
    resolution_clamped=False,
    action=0,           # Action taken (for early termination penalty)
    current_step=5      # Current step (for early termination penalty)
)
```

### RewardNormalizer

```python
from rl_sc_cluster_utils.environment import RewardNormalizer

normalizer = RewardNormalizer(clip_range=10.0)

# Update and normalize
normalizer.update(reward)
normalized = normalizer.normalize(reward)

# Or use convenience method
normalized = normalizer.update_and_normalize(reward)
```

---

## Q_cluster: Clustering Quality

### Components

```python
Q_cluster = 0.5 * silhouette_for_reward + 0.3 * modularity + 0.2 * balance
```

### Silhouette Score (Weight: 0.5)

**Definition**: Measures how well-separated clusters are.

**Computation** (via shared utilities):
```python
from rl_sc_cluster_utils.environment.utils import compute_clustering_quality_metrics

silhouette, modularity, balance = compute_clustering_quality_metrics(
    adata, embeddings, neighbors_computed=True
)
```

**Range**: [-1, 1]
- **+1**: Perfect separation (clusters far apart, cells close within clusters)
- **0**: Overlapping clusters
- **-1**: Poor clustering (cells closer to other clusters than own)

**Silhouette Handling**:
- **Raw silhouette** (`info["silhouette"]`): Preserved in info dict for interpretation (can be negative)
- **Silhouette for reward** (`info["silhouette_for_reward"]`): Shifted by `silhouette_shift` (default: 0.5) to keep non-negative
- **Rationale**: Negative silhouette scores cause RL agents to terminate early. Shifting preserves relative differences while avoiding negative rewards.

**Why 50% weight**: Primary indicator of clustering quality.

---

### Graph Modularity (Weight: 0.3)

**Definition**: Measures quality of community structure in k-NN graph.

**Computation**: Uses `scanpy.metrics.clustering.modularity()` when neighbors graph is available.

**Range**: Typically [-0.5, 1]
- **+1**: Perfect community structure
- **0**: Random assignment
- **Negative**: Worse than random

**Why 30% weight**: Important for graph-based clustering validation.

---

### Cluster Balance (Weight: 0.2)

**Definition**: Measures uniformity of cluster sizes.

**Computation**:
```python
balance = 1 - (std_size / (mean_size + 1e-10))
```

**Range**: [0, 1]
- **1**: Perfectly balanced (all clusters same size)
- **0**: Highly imbalanced (one huge cluster, many tiny ones)

**Why 20% weight**: Prevents over-splitting or over-merging.

---

## Q_GAG: GAG Enrichment Separation

### Overview

Measures how well clusters separate cells by GAG-sulfation pathway expression.

**Implementation**:
```python
# Compute GAG metrics using shared utilities
from rl_sc_cluster_utils.environment.utils import compute_gag_enrichment_metrics

gag_metrics = compute_gag_enrichment_metrics(adata, gene_sets)

# Normalize F-statistics and average
f_stats_normalized = [np.log1p(f_stat) / 10.0 for f_stat in f_stats]
Q_GAG = np.mean(f_stats_normalized)
```

### Per-Cell Enrichment Scores

**Method**: Simplified AUCell (mean expression of gene set per cell)

```python
from rl_sc_cluster_utils.environment.utils import compute_enrichment_scores

scores = compute_enrichment_scores(adata, gene_set)
```

---

### ANOVA F-Statistic

**Definition**: Measures how well clusters separate by enrichment levels.

**Normalization**: `log1p(f_stat) / 10.0` to scale to ~[0, 1]

**Range**: [0, ∞] (raw), [0, ~1] (normalized)
- **High F-stat**: Clusters have very different enrichment levels (good separation)
- **Low F-stat**: Clusters have similar enrichment (poor separation)

---

### Mutual Information

**Definition**: Measures information shared between cluster labels and enrichment levels.

**Computation**: Enrichment scores are binned into 10 bins, then MI is computed.

**Range**: [0, log(n_clusters)]
- **High MI**: Cluster labels predict enrichment well (good separation)
- **Low MI**: No relationship between clusters and enrichment

---

## Penalties

### Implemented Penalty Components

```python
penalty = degenerate_penalty + singleton_penalty + bounds_penalty
```

### Degenerate States Penalty

**Purpose**: Discourage biologically meaningless clusterings.

| Condition | Penalty |
|-----------|---------|
| `n_clusters == 1` | +1.0 |
| `n_clusters > 0.3 * n_cells` | +1.0 |

**Rationale**:
- **1 cluster**: No separation, useless for analysis
- **>30% clusters**: Over-fitting, likely noise

---

### Singleton Penalty

**Purpose**: Discourage clusters with too few cells.

| Condition | Penalty |
|-----------|---------|
| Cluster with < 10 cells | +0.1 per singleton |

**Rationale**: Clusters with very few cells are not biologically meaningful cell types.

---

### Bounds Penalty

**Purpose**: Discourage hitting resolution boundaries.

| Condition | Penalty |
|-----------|---------|
| `resolution_clamped == True` | +0.1 |

**Rationale**: Small signal to learn boundaries without harsh punishment.

---

## Reward Modes

The reward system supports three modes to handle different training scenarios:

### 1. Shaped Mode (Default)

**Purpose**: Keeps rewards non-negative to prevent early termination.

**Formula**:
```python
raw_reward = α * Q_cluster + β * Q_GAG_transformed - δ * Penalty
baseline = running_average(raw_reward)  # Last 100 steps
reward = raw_reward - baseline + 0.1  # Offset ensures mostly positive
```

**When to use**: Default for most training scenarios. Prevents agent from terminating early due to negative rewards.

### 2. Improvement Mode

**Purpose**: Rewards improvement rather than absolute quality.

**Formula**:
```python
current_potential = α * Q_cluster + β * Q_GAG_transformed - δ * Penalty
if first_step:
    reward = current_potential + exploration_bonus
else:
    reward = (current_potential - previous_potential) + exploration_bonus
```

**When to use**: When you want the agent to focus on making progress rather than achieving absolute quality.

### 3. Absolute Mode

**Purpose**: Direct composite reward without shaping.

**Formula**:
```python
reward = α * Q_cluster + β * Q_GAG_transformed - δ * Penalty
```

**When to use**: When you want the raw reward signal without any transformation.

---

## GAG Non-Linear Transformation

**Purpose**: Emphasize GAG-sulfation patterns in the reward signal.

**Transformation** (if `gag_nonlinear=True`):
```python
Q_GAG_transformed = (Q_GAG * gag_scale) ** 2
```

**Default**: `gag_scale=6.0`, `gag_nonlinear=True`

**Rationale**:
- Squaring amplifies differences in GAG enrichment
- Helps agent focus on biologically meaningful GAG-sulfation patterns
- Prevents GAG signal from being overwhelmed by clustering quality metrics

**Example**:
```python
# Raw Q_GAG = 0.1
# Transformed = (0.1 * 6.0) ** 2 = 0.36
# This makes GAG differences more pronounced in the reward
```

---

## Early Termination Penalty

**Purpose**: Discourage Accept action (action 4) before sufficient exploration.

**Logic**:
```python
if action == 4 and current_step < min_steps_before_accept:
    reward = early_termination_penalty  # Default: -5.0
```

**Default**: `early_termination_penalty=-5.0`, `min_steps_before_accept=20`

**Rationale**: Prevents agent from terminating episodes too early without exploring the action space.

---

## Composite Reward

### Final Computation

The reward is computed in `RewardCalculator.compute_reward()`:

```python
def compute_reward(self, adata, action=None, current_step=None, ...):
    # Compute Q_cluster (uses shifted silhouette)
    Q_cluster = 0.5 * silhouette_for_reward + 0.3 * modularity + 0.2 * balance

    # Compute Q_GAG (normalized F-statistics, raw)
    Q_GAG = mean(log1p(f_stat) / 10.0 for each gene set)

    # Apply non-linear transformation if enabled
    if gag_nonlinear:
        Q_GAG_transformed = (Q_GAG * gag_scale) ** 2
    else:
        Q_GAG_transformed = Q_GAG

    # Compute penalty
    penalty = degenerate_penalty + singleton_penalty + bounds_penalty

    # Raw reward
    raw_reward = alpha * Q_cluster + beta * Q_GAG_transformed - delta * penalty

    # Apply reward mode (shaped/improvement/absolute)
    reward = apply_reward_mode(raw_reward, ...)

    # Apply early termination penalty if applicable
    if action == 4 and current_step < min_steps_before_accept:
        reward = early_termination_penalty

    return reward, info
```

---

## Reward Normalization

### RewardNormalizer Class

Implemented in `rl_sc_cluster_utils/environment/rewards.py`:

```python
class RewardNormalizer:
    def __init__(self, clip_range=10.0):
        self.clip_range = clip_range
        self._rewards = []
        self._mean = 0.0
        self._std = 1.0

    def update(self, reward):
        """Update running statistics."""
        self._rewards.append(reward)
        if len(self._rewards) >= 2:
            self._mean = np.mean(self._rewards)
            self._std = np.std(self._rewards)

    def normalize(self, reward):
        """Normalize reward using running statistics."""
        normalized = (reward - self._mean) / (self._std + 1e-10)
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def reset(self):
        """Reset statistics (called on episode reset)."""
        self._rewards = []
        self._mean = 0.0
        self._std = 1.0
```

### Usage in ClusteringEnv

```python
env = ClusteringEnv(
    adata,
    gene_sets=gene_sets,
    normalize_rewards=True  # Enable normalization
)
```

When enabled, reward normalization:
- Resets on each `env.reset()` call
- Updates running statistics on each step
- Returns normalized reward from `env.step()`

---

## Edge Cases

### Single Cluster
- **Silhouette**: Set to 0 (no separation)
- **Modularity**: 0 (no comparison possible)
- **Balance**: Set to 1 (perfectly balanced)
- **Q_GAG**: Set to 0 (no separation possible)
- **Penalty**: +1.0 for degenerate state

### No Clusters
- All metrics default to 0
- Balance defaults to 0

### Singleton Clusters
- **Silhouette**: May be undefined (set to 0)
- **F-stat**: May be NaN (set to 0)
- **Penalty**: +0.1 per cluster with < 10 cells

### Missing Gene Sets
- If `gene_sets` is empty or None: Q_GAG = 0.0
- If gene set has no valid genes: F-stat = 0 for that set

---

## Info Dictionary

The `step()` method returns an info dictionary with reward components:

```python
info = {
    # Reward components
    "raw_reward": ...,          # Reward before normalization (if applicable)
    "reward": ...,               # Final reward value
    "Q_cluster": ...,            # Clustering quality score
    "Q_GAG": ...,                # Raw GAG enrichment score (before transformation)
    "Q_GAG_transformed": ...,    # Transformed GAG score (if gag_nonlinear=True)
    "penalty": ...,              # Total penalty
    "reward_mode": ...,          # Reward mode used ("absolute", "improvement", "shaped")
    "baseline": ...,             # Baseline value (for shaped mode)

    # Individual metrics
    "silhouette": ...,           # Raw silhouette score (preserved for interpretation)
    "silhouette_for_reward": ..., # Silhouette value used in reward computation
    "modularity": ...,
    "balance": ...,
    "n_singletons": ...,
    "mean_f_stat": ...,
    "f_stats": ...,              # Per-gene-set F-statistics

    # Other info
    "n_clusters": ...,
    "resolution_clamped": ...,
    ...
}
```

**Note**: Raw silhouette is always preserved in `info["silhouette"]` for interpretation, even though the reward computation uses the shifted version.

---

## Hyperparameter Tuning

### Default Values (GAG-Focused Configuration)
- α = 0.2 (clustering quality) - Lower weight to emphasize biology
- β = 2.0 (GAG enrichment) - Higher weight for biological signal
- δ = 0.01 (penalty multiplier) - Lower penalty weight
- `reward_mode = "shaped"` - Avoids negative rewards
- `gag_nonlinear = True` - Emphasize GAG patterns
- `gag_scale = 6.0` - Scaling for GAG transformation
- `early_termination_penalty = -5.0` - Discourage early Accept
- `min_steps_before_accept = 20` - Minimum exploration steps

### Custom Configuration

```python
env = ClusteringEnv(
    adata,
    gene_sets=gene_sets,
    reward_mode="shaped",        # Choose reward mode
    reward_alpha=0.2,            # Custom clustering weight
    reward_beta=2.0,             # Custom GAG weight (high for biology)
    reward_delta=0.01,           # Custom penalty weight
    gag_nonlinear=True,          # Enable GAG transformation
    gag_scale=6.0,               # GAG scaling factor
    exploration_bonus=0.2,       # For improvement mode
    silhouette_shift=0.5,        # Silhouette shift amount
    early_termination_penalty=-5.0,  # Early Accept penalty
    min_steps_before_accept=20,      # Minimum steps
)
```

### Migration from Wrapper

If you were using `DeltaRewardWrapper`, migrate to the consolidated system:

**Before (with wrapper)**:
```python
env = ClusteringEnv(adata, gene_sets, reward_alpha=0.2, reward_beta=2.0, reward_delta=0.01)
env = DeltaRewardWrapper(env)
```

**After (consolidated)**:
```python
env = ClusteringEnv(
    adata,
    gene_sets,
    reward_mode="shaped",  # or "improvement" for delta rewards
    reward_alpha=0.2,
    reward_beta=2.0,
    reward_delta=0.01,
    gag_nonlinear=True,
    gag_scale=6.0,
    early_termination_penalty=-5.0,
    min_steps_before_accept=20,
)
# No wrapper needed!
```

### Tuning Strategy
- Use Optuna for hyperparameter optimization
- Grid search or Bayesian optimization
- Optimize on validation set (separate from training)

---

---

## Summary

The reward system has been consolidated to support multiple reward modes, GAG-focused configurations, and early termination penalties. Key features:

1. **Three reward modes**: Absolute, improvement, and shaped (default)
2. **GAG non-linear transformation**: Emphasizes biological patterns
3. **Raw silhouette preservation**: Always available in info dict for interpretation
4. **Early termination penalty**: Prevents premature Accept actions
5. **Configurable parameters**: All aspects of reward computation are configurable

**Document Version**: 3.0
**Last Updated**: December 2025
**Implementation Status**: Complete (Stage 5 - Consolidated Reward System)
