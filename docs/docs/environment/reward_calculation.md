# Reward Calculation

This document details the composite reward function used in the clustering environment.

**Implementation Status**: Complete (Stage 4)

---

## Overview

The reward function balances three objectives:
1. **Clustering Quality** (Q_cluster): How well-separated and balanced clusters are
2. **GAG Enrichment** (Q_GAG): How well clusters separate by GAG-sulfation expression
3. **Penalties**: Discourage degenerate states and bad actions

**Formula**:
```python
reward = α * Q_cluster + β * Q_GAG - δ * Penalty
```

**Default weights**: α = 0.6, β = 0.4, δ = 1.0

---

## Implementation

The reward system is implemented in `rl_sc_cluster_utils/environment/rewards.py` with two main classes:

### RewardCalculator

```python
from rl_sc_cluster_utils.environment import RewardCalculator

calculator = RewardCalculator(
    adata,
    gene_sets,
    alpha=0.6,   # Weight for clustering quality
    beta=0.4,    # Weight for GAG enrichment
    delta=1.0    # Weight for penalties
)

# Compute reward
reward, info = calculator.compute_reward(
    adata,
    previous_reward=None,
    resolution_clamped=False
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
Q_cluster = 0.5 * silhouette + 0.3 * modularity + 0.2 * balance
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

## Composite Reward

### Final Computation

The reward is computed in `RewardCalculator.compute_reward()`:

```python
def compute_reward(self, adata, previous_reward=None, resolution_clamped=False):
    # Compute Q_cluster
    Q_cluster = 0.5 * silhouette + 0.3 * modularity + 0.2 * balance

    # Compute Q_GAG (normalized F-statistics)
    Q_GAG = mean(log1p(f_stat) / 10.0 for each gene set)

    # Compute penalty
    penalty = degenerate_penalty + singleton_penalty + bounds_penalty

    # Composite reward
    reward = alpha * Q_cluster + beta * Q_GAG - delta * penalty

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
    "raw_reward": ...,      # Reward before normalization
    "Q_cluster": ...,       # Clustering quality score
    "Q_GAG": ...,           # GAG enrichment score
    "penalty": ...,         # Total penalty

    # Individual metrics
    "silhouette": ...,
    "modularity": ...,
    "balance": ...,
    "n_singletons": ...,
    "mean_f_stat": ...,

    # Other info
    "n_clusters": ...,
    "resolution_clamped": ...,
    ...
}
```

---

## Hyperparameter Tuning

### Default Values
- α = 0.6 (clustering quality)
- β = 0.4 (GAG enrichment)
- δ = 1.0 (penalty multiplier)

### Custom Weights

```python
env = ClusteringEnv(
    adata,
    gene_sets=gene_sets,
    reward_alpha=0.5,   # Custom clustering weight
    reward_beta=0.3,    # Custom GAG weight
    reward_delta=0.8    # Custom penalty weight
)
```

### Tuning Strategy
- Use Optuna for hyperparameter optimization
- Grid search or Bayesian optimization
- Optimize on validation set (separate from training)

---

**Document Version**: 2.0
**Last Updated**: December 2025
**Implementation Status**: Complete (Stage 4)
