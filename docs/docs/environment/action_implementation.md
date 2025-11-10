# Action Implementation

This document details the implementation of the 5 discrete actions in the clustering environment.

---

## Action Space

**Type**: `gymnasium.spaces.Discrete(5)`

**Actions**:
- 0: Split worst cluster
- 1: Merge closest pair
- 2: Re-cluster resolution +0.1
- 3: Re-cluster resolution -0.1
- 4: Accept (terminate episode)

---

## Action 0: Split Worst Cluster

### Objective
Identify the cluster with lowest quality (worst silhouette) and split it into sub-clusters.

### Algorithm

1. **Identify Worst Cluster**:
   ```python
   def find_worst_cluster(adata):
       """Find cluster with lowest mean silhouette."""
       cluster_silhouettes = {}
       for cluster_id in adata.obs['clusters'].unique():
           cluster_mask = adata.obs['clusters'] == cluster_id
           if cluster_mask.sum() < 2:
               continue  # Skip singletons
           cluster_embeddings = adata.obsm['X_scvi'][cluster_mask]
           sil = silhouette_score(cluster_embeddings,
                                  adata.obs['clusters'][cluster_mask])
           cluster_silhouettes[cluster_id] = sil

       worst_cluster = min(cluster_silhouettes, key=cluster_silhouettes.get)
       return worst_cluster
   ```

2. **Extract Subgraph**:
   ```python
   def extract_cluster_subgraph(adata, cluster_id):
       """Extract k-NN subgraph for a cluster."""
       cluster_mask = adata.obs['clusters'] == cluster_id
       cluster_cells = adata.obs_names[cluster_mask]

       # Get neighbors graph (sparse matrix)
       neighbors = adata.uns['neighbors']['connectivities']

       # Extract subgraph (cells in cluster)
       subgraph = neighbors[cluster_mask, :][:, cluster_mask]

       return subgraph, cluster_mask
   ```

3. **Sub-cluster**:
   ```python
   def subcluster(adata, cluster_id, resolution):
       """Run Leiden on cluster subgraph."""
       subgraph, cluster_mask = extract_cluster_subgraph(adata, cluster_id)

       # Create temporary AnnData for sub-clustering
       cluster_adata = adata[cluster_mask].copy()
       cluster_adata.uns['neighbors'] = {'connectivities': subgraph}

       # Run Leiden at higher resolution
       sc.tl.leiden(cluster_adata, resolution=resolution, key_added='subclusters')

       # Map sub-clusters back to original cluster
       new_labels = cluster_adata.obs['subclusters'].values
       return new_labels, cluster_mask
   ```

4. **Update Labels**:
   ```python
   def update_labels_split(adata, cluster_id, new_labels, cluster_mask):
       """Replace cluster with new sub-clusters."""
       # Get current max cluster ID
       max_cluster_id = adata.obs['clusters'].max()

       # Assign new cluster IDs
       new_cluster_ids = new_labels + max_cluster_id + 1

       # Update labels
       adata.obs.loc[cluster_mask, 'clusters'] = new_cluster_ids
   ```

### Edge Cases

- **Singleton cluster**: Cannot split (skip or return error)
- **Only 1 cluster**: Cannot split (skip or return error)
- **Sub-clustering produces 1 cluster**: Keep original cluster (no change)

### Resolution for Sub-clustering
```python
subcluster_resolution = current_resolution + 0.2
```

---

## Action 1: Merge Closest Pair

### Objective
Find the two clusters with minimum centroid distance and merge them.

### Algorithm

1. **Compute Cluster Centroids**:
   ```python
   def compute_centroids(adata):
       """Compute centroid of each cluster in embedding space."""
       centroids = {}
       for cluster_id in adata.obs['clusters'].unique():
           cluster_mask = adata.obs['clusters'] == cluster_id
           cluster_embeddings = adata.obsm['X_scvi'][cluster_mask]
           centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)
       return centroids
   ```

2. **Find Closest Pair**:
   ```python
   def find_closest_clusters(centroids):
       """Find pair of clusters with minimum distance."""
       min_dist = np.inf
         closest_pair = None

       cluster_ids = list(centroids.keys())
       for i, c1 in enumerate(cluster_ids):
           for c2 in cluster_ids[i+1:]:
               dist = np.linalg.norm(centroids[c1] - centroids[c2])
               if dist < min_dist:
                   min_dist = dist
                   closest_pair = (c1, c2)

       return closest_pair
   ```

3. **Merge Clusters**:
   ```python
   def merge_clusters(adata, cluster1, cluster2):
       """Merge two clusters into one."""
       # Use smaller cluster ID as new label
       new_label = min(cluster1, cluster2)

       # Update labels
       mask1 = adata.obs['clusters'] == cluster1
       mask2 = adata.obs['clusters'] == cluster2

       adata.obs.loc[mask1, 'clusters'] = new_label
       adata.obs.loc[mask2, 'clusters'] = new_label
   ```

### Edge Cases

- **Only 1 cluster**: Cannot merge (skip or return error)
- **Tie in distance**: Use first pair found (deterministic)

### Distance Metric
- **Default**: Euclidean distance in embedding space
- **Future**: Allow configuration (cosine, Manhattan, etc.)

---

## Action 2: Re-cluster Resolution +0.1

### Objective
Increase clustering resolution by 0.1 and re-run Leiden on full graph.

### Algorithm

1. **Update Resolution**:
   ```python
   def increment_resolution(current_resolution, max_resolution=2.0):
       """Increment resolution with clamping."""
       new_resolution = min(max_resolution, current_resolution + 0.1)
       clamped = (new_resolution == max_resolution and
                  current_resolution < max_resolution)
       return new_resolution, clamped
   ```

2. **Re-cluster**:
   ```python
   def recluster(adata, resolution):
       """Run Leiden clustering with new resolution."""
       sc.tl.leiden(adata, resolution=resolution, key_added='clusters')
   ```

3. **Apply Penalty if Clamped**:
   ```python
   if clamped:
       penalty = -0.1
   ```

### Edge Cases

- **Already at max resolution**: Clamp and apply penalty
- **Resolution exactly at bound**: No change, apply penalty

---

## Action 3: Re-cluster Resolution -0.1

### Objective
Decrease clustering resolution by 0.1 and re-run Leiden on full graph.

### Algorithm

1. **Update Resolution**:
   ```python
   def decrement_resolution(current_resolution, min_resolution=0.1):
       """Decrement resolution with clamping."""
       new_resolution = max(min_resolution, current_resolution - 0.1)
       clamped = (new_resolution == min_resolution and
                  current_resolution > min_resolution)
       return new_resolution, clamped
   ```

2. **Re-cluster**: Same as Action 2

3. **Apply Penalty if Clamped**: Same as Action 2

---

## Action 4: Accept

### Objective
Terminate the episode with current clustering.

### Algorithm

```python
def accept_action():
    """Terminate episode."""
    terminated = True
    # No changes to clustering
    # Final reward computed with current state
```

### Implementation
- Handled in `step()` method
- Sets `terminated = True`
- No changes to clustering state

---

## Action Validation

### Pre-action Checks

```python
def validate_action(adata, action, current_resolution):
    """Validate action is legal."""
    n_clusters = len(adata.obs['clusters'].unique())

    if action == 0:  # Split
        if n_clusters == 1:
            return False, "Cannot split: only 1 cluster"
        # Check for non-singleton clusters
        ...

    elif action == 1:  # Merge
        if n_clusters == 1:
            return False, "Cannot merge: only 1 cluster"

    elif action == 2:  # Increase resolution
        if current_resolution >= 2.0:
            return False, "Already at max resolution"

    elif action == 3:  # Decrease resolution
        if current_resolution <= 0.1:
            return False, "Already at min resolution"

    return True, None
```

### Post-action Validation

```python
def validate_clustering(adata):
    """Validate clustering is still valid after action."""
    # Check all cells have cluster labels
    assert adata.obs['clusters'].notna().all()

    # Check at least 1 cluster exists
    assert len(adata.obs['clusters'].unique()) >= 1

    # Check no negative cluster IDs
    assert (adata.obs['clusters'] >= 0).all()
```

---

## Action Execution Flow

```python
def execute_action(adata, action, current_resolution):
    """Execute action and return new state."""

    # Validate action
    is_valid, error_msg = validate_action(adata, action, current_resolution)
    if not is_valid:
        # Return same state, negative reward
        return adata, current_resolution, -1.0, error_msg

    # Execute action
    if action == 0:
        adata, new_resolution = split_worst_cluster(adata, current_resolution)
    elif action == 1:
        adata, new_resolution = merge_closest_pair(adata, current_resolution)
    elif action == 2:
        adata, new_resolution, clamped = increment_resolution(adata, current_resolution)
    elif action == 3:
        adata, new_resolution, clamped = decrement_resolution(adata, current_resolution)
    elif action == 4:
        # Accept - no changes
        new_resolution = current_resolution

    # Validate result
    validate_clustering(adata)

    return adata, new_resolution, None, None
```

---

## Performance Considerations

### Caching
- Cache cluster centroids (recompute only after merge/split)
- Cache subgraphs (reuse for multiple splits)
- Cache resolution (only update on re-cluster actions)

### Optimization
- Vectorize centroid computation
- Use sparse matrices for subgraph extraction
- Batch Leiden clustering when possible

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
