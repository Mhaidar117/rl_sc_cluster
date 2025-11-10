# State Representation

This document details the 35-dimensional state vector used in the clustering environment.

---

## Overview

The state vector encodes the current clustering configuration using:
- **Global metrics** (3 dims): Cluster count and size distribution
- **Quality metrics** (3 dims): Clustering quality measures
- **GAG enrichment** (28 dims): Biological signal separation
- **Progress** (1 dim): Episode progress tracking

**Total**: 35 dimensions

---

## State Vector Structure

### Index 0-2: Global Metrics (3 dimensions)

#### Index 0: Normalized Cluster Count
```python
n_clusters / n_cells
```
- **Range**: [0, 1] (theoretically, but typically [0.01, 0.3])
- **Meaning**: Proportion of cells that are cluster centers (inverse of mean cluster size)
- **Computation**: `len(adata.obs['clusters'].unique()) / adata.n_obs`

#### Index 1: Normalized Mean Cluster Size
```python
mean_cluster_size / n_cells
```
- **Range**: [0, 1]
- **Meaning**: Average cluster size as proportion of total cells
- **Computation**: `adata.obs['clusters'].value_counts().mean() / adata.n_obs`

#### Index 2: Cluster Size Entropy
```python
H = -Σ p_i * log(p_i)
```
- **Range**: [0, log(n_clusters)]
- **Meaning**: Entropy of cluster size distribution (higher = more balanced)
- **Computation**:
  ```python
  cluster_sizes = adata.obs['clusters'].value_counts()
  p = cluster_sizes / cluster_sizes.sum()
  entropy = -np.sum(p * np.log(p + 1e-10))
  ```

---

### Index 3-5: Quality Metrics (3 dimensions)

#### Index 3: Silhouette Score
```python
silhouette_score(embeddings, cluster_labels)
```
- **Range**: [-1, 1]
- **Meaning**: How well-separated clusters are (higher = better separation)
- **Computation**: `sklearn.metrics.silhouette_score(adata.obsm['X_scvi'], adata.obs['clusters'])`

#### Index 4: Graph Modularity
```python
modularity(neighbors_graph, cluster_labels)
```
- **Range**: [0, 1] (typically [-0.5, 1])
- **Meaning**: Quality of community structure in k-NN graph
- **Computation**: `scanpy.tl.modularity(adata, key='clusters')` or custom implementation

#### Index 5: Cluster Balance
```python
1 - (std_size / mean_size)
```
- **Range**: [0, 1] (higher = more balanced)
- **Meaning**: Uniformity of cluster sizes
- **Computation**:
  ```python
  cluster_sizes = adata.obs['clusters'].value_counts()
  balance = 1 - (cluster_sizes.std() / (cluster_sizes.mean() + 1e-10))
  ```

---

### Index 6-33: GAG Enrichment (28 dimensions = 7 gene sets × 4 metrics)

For each of 7 GAG-sulfation gene sets, compute 4 metrics:

#### Per Gene Set (4 dimensions each):

**Index 6 + 4*i + 0: Mean Enrichment**
```python
mean(AUCell_scores) across all clusters
```
- **Range**: [0, 1] (AUCell scores)
- **Meaning**: Average enrichment across clusters
- **Computation**: `np.mean([cluster_mean_enrichment for cluster in clusters])`

**Index 6 + 4*i + 1: Max Enrichment**
```python
max(cluster_mean_enrichment) across clusters
```
- **Range**: [0, 1]
- **Meaning**: Maximum cluster enrichment (identifies GAG-high cluster)
- **Computation**: `np.max([cluster_mean_enrichment for cluster in clusters])`

**Index 6 + 4*i + 2: ANOVA F-Statistic**
```python
f_oneway(enrichment ~ cluster)
```
- **Range**: [0, ∞]
- **Meaning**: How well clusters separate by enrichment (higher = better separation)
- **Computation**: `scipy.stats.f_oneway(*[enrichment[clusters==c] for c in unique_clusters])`

**Index 6 + 4*i + 3: Mutual Information**
```python
mutual_info_score(cluster_labels, enrichment_bins)
```
- **Range**: [0, log(n_clusters)]
- **Meaning**: Information shared between clusters and enrichment levels
- **Computation**:
  ```python
  enrichment_bins = pd.cut(enrichment_scores, bins=10, labels=False)
  mi = sklearn.metrics.mutual_info_score(cluster_labels, enrichment_bins)
  ```

#### Gene Set Order (7 sets):
1. CS biosynthesis (i=0, indices 6-9)
2. CS sulfation (i=1, indices 10-13)
3. HS biosynthesis (i=2, indices 14-17)
4. HS sulfation (i=3, indices 18-21)
5. Sulfate activation (i=4, indices 22-25)
6. PNN core (i=5, indices 26-29)
7. Additional set (i=6, indices 30-33)

---

### Index 34: Progress (1 dimension)

#### Index 34: Normalized Episode Progress
```python
current_step / max_steps
```
- **Range**: [0, 1]
- **Meaning**: How far through the episode (0 = start, 1 = end)
- **Computation**: `self.current_step / self.max_steps`

---

## Enrichment Score Computation

### AUCell Method
For each gene set, compute per-cell enrichment scores:

```python
def compute_aucell_scores(adata, gene_set):
    """
    Compute AUCell enrichment scores for a gene set.

    AUCell: Area Under the Curve of gene expression ranking
    Higher score = cell expresses more genes in the set
    """
    # Get expression matrix (log-normalized)
    expr = adata.X  # or adata.layers['log_normalized']

    # Get gene indices for gene set
    gene_indices = [adata.var_names.get_loc(gene) for gene in gene_set]

    # Rank genes per cell (higher expression = higher rank)
    ranks = np.argsort(expr[:, gene_indices], axis=1)

    # Compute AUC: proportion of gene set genes in top-k
    # (Simplified - actual AUCell is more complex)
    aucell_scores = np.mean(ranks, axis=1)

    return aucell_scores
```

---

## Normalization

### Optional State Normalization
When `normalize_state=True`, apply min-max scaling per dimension:

```python
def normalize_state(state, state_min, state_max):
    """Normalize state to [0, 1] range."""
    return (state - state_min) / (state_max - state_min + 1e-10)
```

**Normalization ranges** (estimated):
- Global metrics: [0, 1] (already normalized)
- Quality metrics: [-1, 1] for silhouette, [0, 1] for modularity/balance
- GAG enrichment: [0, 1] for mean/max, [0, 10] for F-stat, [0, log(7)] for MI
- Progress: [0, 1] (already normalized)

---

## Caching Strategy

### What to Cache

**Permanent (set at init)**:
- Embeddings: `adata.obsm['X_scvi']`
- Graph structure: `adata.uns['neighbors']`
- Gene sets: List of gene sets

**Episode (reset each episode)**:
- Current clustering: `adata.obs['clusters']`
- Current resolution: `self.current_resolution`

**Step (update each step)**:
- Previous state vector: `self._cached_state`
- Previous reward: `self._cached_reward`

### When to Recompute

- **Global metrics**: After any action that changes cluster count
- **Quality metrics**: After any action that changes clustering
- **GAG enrichment**: After any action that changes clustering
- **Progress**: Every step (always recompute)

---

## Edge Cases

### Single Cluster (n_clusters == 1)
- **Entropy**: Set to 0 (no entropy with 1 cluster)
- **Silhouette**: Set to 0 (no separation)
- **Modularity**: Compute normally (may be high if graph has structure)
- **Balance**: Set to 1 (perfectly balanced - 1 cluster)

### Singleton Clusters (n_clusters == n_cells)
- **Entropy**: Maximum (log(n_cells))
- **Silhouette**: May be undefined (use 0 or skip)
- **Modularity**: Low (no community structure)
- **Balance**: 1 (all clusters same size)

### Missing Gene Sets
- If gene set has no expressed genes: Set all 4 metrics to 0
- If gene set has <3 genes: Warn and set metrics to 0

---

## Validation

### State Vector Validation
```python
def validate_state(state):
    """Validate state vector has correct shape and ranges."""
    assert state.shape == (35,), f"Expected shape (35,), got {state.shape}"
    assert np.all(np.isfinite(state)), "State contains non-finite values"
    # Check ranges (warn if outside expected)
    ...
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
