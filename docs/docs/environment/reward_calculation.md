# Reward Calculation

This document details the composite reward function used in the clustering environment.

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

## Q_cluster: Clustering Quality

### Components

```python
Q_cluster = 0.5 * silhouette + 0.3 * modularity + 0.2 * balance
```

### Silhouette Score (Weight: 0.5)

**Definition**: Measures how well-separated clusters are.

**Computation**:
```python
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(
    adata.obsm['X_scvi'],  # Embeddings
    adata.obs['clusters']  # Cluster labels
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

**Computation**:
```python
import scanpy as sc

sc.tl.modularity(adata, key='clusters')
modularity = adata.uns['modularity']['clusters']
```

**Or custom implementation**:
```python
def compute_modularity(neighbors_graph, cluster_labels):
    """Compute Newman-Girvan modularity."""
    # Modularity = (fraction of edges within communities) -
    #              (expected fraction if random)
    ...
```

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
cluster_sizes = adata.obs['clusters'].value_counts()
mean_size = cluster_sizes.mean()
std_size = cluster_sizes.std()

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

**Computation**:
```python
Q_GAG = mean([F_statistic + mutual_info for gene_set in gene_sets])
```

For each of 7 gene sets, compute:
1. **ANOVA F-statistic**: How well clusters separate by enrichment
2. **Mutual Information**: Information shared between clusters and enrichment

---

### Per-Cell Enrichment Scores

**Method**: AUCell (Area Under the Curve)

```python
def compute_aucell_scores(adata, gene_set):
    """
    Compute AUCell enrichment scores for a gene set.

    For each cell:
    1. Rank all genes by expression (highest = rank 1)
    2. Compute AUC: area under curve of gene set genes in ranking
    3. Higher AUC = cell expresses more genes in set
    """
    # Get expression matrix (log-normalized)
    expr = adata.X  # or adata.layers['log_normalized']

    # Get gene indices for gene set
    gene_indices = [adata.var_names.get_loc(gene)
                    for gene in gene_set if gene in adata.var_names]

    if len(gene_indices) == 0:
        return np.zeros(adata.n_obs)  # No genes found

    # Rank genes per cell (higher expression = higher rank)
    # Simplified: use mean expression of gene set
    aucell_scores = np.mean(expr[:, gene_indices], axis=1)

    return aucell_scores
```

**Alternative**: Use `scanpy.tl.score_genes` or `scanpy.tl.score_genes_cell_cycle`

---

### ANOVA F-Statistic

**Definition**: Measures how well clusters separate by enrichment levels.

**Computation**:
```python
from scipy.stats import f_oneway

def compute_f_statistic(adata, gene_set):
    """Compute ANOVA F-statistic for enrichment ~ cluster."""
    enrichment = compute_aucell_scores(adata, gene_set)

    # Group by cluster
    clusters = adata.obs['clusters'].unique()
    groups = [enrichment[adata.obs['clusters'] == c] for c in clusters]

    # One-way ANOVA
    f_stat, p_value = f_oneway(*groups)

    return f_stat
```

**Range**: [0, ∞]
- **High F-stat**: Clusters have very different enrichment levels (good separation)
- **Low F-stat**: Clusters have similar enrichment (poor separation)

---

### Mutual Information

**Definition**: Measures information shared between cluster labels and enrichment levels.

**Computation**:
```python
from sklearn.metrics import mutual_info_score

def compute_mutual_info(adata, gene_set):
    """Compute MI between clusters and enrichment."""
    enrichment = compute_aucell_scores(adata, gene_set)

    # Discretize enrichment into bins
    enrichment_bins = pd.cut(enrichment, bins=10, labels=False)

    # Compute MI
    mi = mutual_info_score(adata.obs['clusters'], enrichment_bins)

    return mi
```

**Range**: [0, log(n_clusters)]
- **High MI**: Cluster labels predict enrichment well (good separation)
- **Low MI**: No relationship between clusters and enrichment

---

### Q_GAG Aggregation

```python
def compute_q_gag(adata, gene_sets):
    """Compute Q_GAG across all gene sets."""
    gag_scores = []

    for gene_set_name, gene_set in gene_sets.items():
        # Compute enrichment scores
        enrichment = compute_aucell_scores(adata, gene_set)

        # Compute F-statistic
        f_stat = compute_f_statistic(adata, gene_set)

        # Compute mutual information
        mi = compute_mutual_info(adata, gene_set)

        # Combine (both measure separation, so sum)
        gag_scores.append(f_stat + mi)

    # Average across gene sets
    Q_GAG = np.mean(gag_scores)

    return Q_GAG
```

---

## Penalties

### Penalty Components

```python
penalty = degenerate_penalty + degradation_penalty + bounds_penalty
```

---

### Degenerate States Penalty

**Purpose**: Discourage biologically meaningless clusterings.

```python
def compute_degenerate_penalty(adata):
    """Compute penalty for degenerate clustering states."""
    penalty = 0
    n_clusters = len(adata.obs['clusters'].unique())
    n_cells = adata.n_obs

    # Too few clusters (all cells in one cluster)
    if n_clusters == 1:
        penalty += 5

    # Too many clusters (over-splitting)
    if n_clusters > 0.3 * n_cells:
        penalty += 5

    # Singleton clusters (biologically meaningless)
    cluster_sizes = adata.obs['clusters'].value_counts()
    n_singletons = (cluster_sizes == 1).sum()
    penalty += n_singletons * 2

    return penalty
```

**Rationale**:
- **1 cluster**: No separation, useless for analysis
- **>30% clusters**: Over-fitting, likely noise
- **Singletons**: Not biologically meaningful cell types

---

### Quality Degradation Penalty

**Purpose**: Discourage actions that worsen clustering.

```python
def compute_degradation_penalty(previous_reward, current_reward):
    """Penalize actions that decrease reward."""
    if previous_reward is None:
        return 0

    if current_reward < previous_reward:
        # Penalty proportional to degradation
        degradation = previous_reward - current_reward
        return degradation
    else:
        return 0
```

**Rationale**: Guide agent away from bad actions.

---

### Bounds Penalty

**Purpose**: Discourage hitting resolution boundaries.

```python
def compute_bounds_penalty(resolution_clamped):
    """Small penalty for hitting resolution bounds."""
    if resolution_clamped:
        return 0.1
    else:
        return 0
```

**Rationale**: Small signal to learn boundaries without harsh punishment.

---

## Composite Reward

### Final Computation

```python
def compute_reward(
    adata,
    gene_sets,
    previous_reward=None,
    resolution_clamped=False,
    alpha=0.6,
    beta=0.4,
    delta=1.0
):
    """Compute composite reward."""

    # Clustering quality
    Q_cluster = compute_q_cluster(adata)

    # GAG enrichment
    Q_GAG = compute_q_gag(adata, gene_sets)

    # Penalties
    degenerate = compute_degenerate_penalty(adata)
    degradation = compute_degradation_penalty(previous_reward, current_reward)
    bounds = compute_bounds_penalty(resolution_clamped)

    penalty = degenerate + degradation + bounds

    # Composite reward
    reward = alpha * Q_cluster + beta * Q_GAG - delta * penalty

    return reward
```

---

## Reward Normalization (Optional)

### Running Statistics

```python
class RewardNormalizer:
    def __init__(self):
        self.reward_history = []
        self.mean = 0.0
        self.std = 1.0

    def update(self, reward):
        """Update running statistics."""
        self.reward_history.append(reward)

        # Update mean and std
        self.mean = np.mean(self.reward_history)
        self.std = np.std(self.reward_history) + 1e-10

    def normalize(self, reward):
        """Normalize reward using running statistics."""
        return (reward - self.mean) / self.std
```

### Usage

```python
if normalize_rewards:
    normalized_reward = reward_normalizer.normalize(reward)
else:
    normalized_reward = reward
```

**When to update**: After each episode (not each step).

---

## Edge Cases

### Single Cluster
- **Silhouette**: Set to 0 (no separation)
- **Modularity**: Compute normally (may be high)
- **Balance**: Set to 1 (perfectly balanced)
- **Q_GAG**: Set to 0 (no separation possible)

### Singleton Clusters
- **Silhouette**: May be undefined (set to 0)
- **Modularity**: Low (no community structure)
- **Balance**: 1 (all same size, but meaningless)

### Missing Gene Sets
- If gene set has no genes: Set F-stat and MI to 0
- If gene set has <3 genes: Warn and set to 0

---

## Hyperparameter Tuning

### Default Values
- α = 0.6 (clustering quality)
- β = 0.4 (GAG enrichment)
- δ = 1.0 (penalty multiplier)

### Tuning Strategy
- Use Optuna for hyperparameter optimization
- Grid search or Bayesian optimization
- Optimize on validation set (separate from training)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
