# Environment Design Decisions

This document records key design decisions made during environment development, along with rationale and alternatives considered.

---

## Table of Contents

1. [State Normalization](#state-normalization)
2. [Reward Normalization](#reward-normalization)
3. [Resolution Bounds Handling](#resolution-bounds-handling)
4. [Penalty Mechanisms](#penalty-mechanisms)
5. [Caching Strategy](#caching-strategy)
6. [Action Masking](#action-masking)

---

## State Normalization

### Decision
Make state normalization **optional** via parameter `normalize_state` (default: `False`).

### Rationale
- **Research flexibility**: Enables comparison of normalized vs. non-normalized performance
- **Biological interpretability**: Raw values may carry meaningful scale information
- **Algorithm compatibility**: Some algorithms (e.g., PPO) handle unnormalized states well

### Implementation
- When `normalize_state=True`: Apply min-max scaling per dimension
- Preserve original values for comparison studies
- Normalization computed on training data, applied consistently

### Alternatives Considered
- **Always normalize**: Rejected - limits research flexibility
- **Never normalize**: Rejected - may hurt learning in some cases
- **Z-score normalization**: Considered but min-max chosen for bounded output

---

## Reward Normalization

### Decision
Make reward normalization **optional** via parameter `normalize_rewards` (default: `False`).

### Rationale
- **PPO stability**: Normalized rewards can stabilize policy gradients
- **Hyperparameter transfer**: Makes α, β, δ more transferable across datasets
- **Research comparison**: Allows comparison of normalized vs. raw rewards

### Implementation
- When `normalize_rewards=True`: Use running mean and standard deviation
- Update normalization stats after each episode
- Formula: `(reward - mean) / std`

### Alternatives Considered
- **Always normalize**: Rejected - may obscure biological reward magnitudes
- **Never normalize**: Rejected - may cause training instability
- **Reward clipping**: Considered but normalization preferred for stability

---

## Resolution Bounds Handling

### Decision
**Clamp resolution** to [0.1, 2.0] with **small penalty (-0.1)** when hitting bounds.

### Rationale
- **Simplicity**: Clamping is straightforward to implement
- **Learning signal**: Small penalty teaches agent boundaries without harsh punishment
- **PPO compatibility**: Clipping is standard in PPO, agent should learn boundaries

### Implementation
```python
if action == 2:  # Increase resolution
    new_resolution = min(2.0, current_resolution + 0.1)
    penalty = -0.1 if new_resolution == 2.0 else 0.0
elif action == 3:  # Decrease resolution
    new_resolution = max(0.1, current_resolution - 0.1)
    penalty = -0.1 if new_resolution == 0.1 else 0.0
```

### Risk: Agent Getting Stuck
- **Concern**: Agent might repeatedly hit bounds, getting stuck in local optima
- **Mitigation**:
  - Small penalty discourages but doesn't prevent boundary exploration
  - Entropy regularization in PPO encourages exploration
  - Monitor action distribution during training
  - Future: Implement action masking to disable invalid actions

### Alternatives Considered
- **No clamping**: Rejected - invalid resolutions would cause errors
- **Large penalty**: Rejected - too harsh, prevents learning
- **Action masking**: Deferred to future enhancement (cleaner signal)

---

## Penalty Mechanisms

### Decision
Use **composite penalty system** with three components:

1. **Degenerate States**: Penalize biologically meaningless clusterings
2. **Quality Degradation**: Penalize actions that worsen clustering
3. **Boundary Violations**: Penalize hitting resolution bounds

### Rationale
- **Biological validity**: Ensures clusters are meaningful
- **Learning signal**: Guides agent away from bad states
- **Balanced**: Multiple penalty types prevent over-penalization

### Implementation
```python
penalty = 0

# Degenerate states
if n_clusters == 1:
    penalty += 5
if n_clusters > 0.3 * n_cells:
    penalty += 5
penalty += n_singletons * 2

# Quality degradation
if previous_reward and current_reward < previous_reward:
    penalty += max(0, previous_reward - current_reward)

# Boundary violations
if resolution_clamped:
    penalty += 0.1
```

### Alternatives Considered
- **Single penalty type**: Rejected - too simplistic
- **No degradation penalty**: Rejected - agent might oscillate
- **Harsh penalties**: Rejected - prevents exploration

---

## Caching Strategy

### Decision
Implement **three-level caching system**:

1. **Permanent cache**: Embeddings, graph structure, gene sets (set at init)
2. **Episode cache**: Current clustering, resolution (reset each episode)
3. **Step cache**: Previous state values (update each step)

### Rationale
- **Performance**: Reduces redundant computations
- **Correctness**: Only recompute metrics affected by last action
- **Memory efficiency**: Cache only what's needed

### Implementation
```python
# Permanent (set once)
self._cached_embeddings = adata.obsm['X_scvi']
self._cached_graph = adata.uns['neighbors']
self._cached_gene_sets = gene_sets

# Episode (reset each episode)
self._cached_clustering = None
self._cached_resolution = None

# Step (update each step)
self._cached_state = None
self._cached_reward = None
```

### Alternatives Considered
- **No caching**: Rejected - too slow for RL training
- **Cache everything**: Rejected - memory inefficient
- **Lazy evaluation**: Considered but eager caching preferred for clarity

---

## Action Masking

### Decision
**Defer to future enhancement** - not implemented in initial version.

### Rationale
- **Complexity**: Adds implementation complexity
- **Current solution sufficient**: Clamping + penalty works for now
- **Future optimization**: Can add if agent gets stuck at boundaries

### Future Implementation
```python
def get_action_mask(self):
    """Return mask for valid actions."""
    mask = np.ones(5, dtype=bool)
    
    if self.current_resolution >= 2.0:
        mask[2] = False  # Can't increase resolution
    if self.current_resolution <= 0.1:
        mask[3] = False  # Can't decrease resolution
    
    return mask
```

---

## Terminal State Handling

### Decision
**Accept action (4) terminates episode** and includes **final reward evaluation**.

### Rationale
- **Clear termination**: Agent explicitly chooses to stop
- **Final evaluation**: Terminal state gets final reward for that clustering
- **PPO compatibility**: Standard episodic termination

### Implementation
```python
if action == 4:  # Accept
    terminated = True
    # Final reward computed with current clustering
    reward = self._compute_final_reward()
```

---

## Validation Metrics

### Decision
Implement **independent scoring function** separate from environment reward.

### Rationale
- **Unbiased evaluation**: Not influenced by reward function design
- **Research validation**: Compare against ground truth or baselines
- **Reproducibility**: Standard metrics for comparison

### Implementation
```python
def compute_validation_metrics(adata, gene_sets):
    """Compute metrics independent of reward function."""
    return {
        'silhouette': silhouette_score(...),
        'modularity': modularity(...),
        'gag_separation': compute_gag_separation(...),
        'n_clusters': len(adata.obs['clusters'].unique())
    }
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX

