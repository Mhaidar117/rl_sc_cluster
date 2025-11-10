# RL Environment Development Plan

**Project**: RL-Guided Refinement of scRNA-seq Clustering  
**Component**: Gymnasium-Compatible RL Environment  
**Status**: Planning Phase → Stage 1 Implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Development Stages](#development-stages)
3. [Stage 1: Minimal Gymnasium Environment](#stage-1-minimal-gymnasium-environment)
4. [Stage 2: State Representation](#stage-2-state-representation)
5. [Stage 3: Action Implementation](#stage-3-action-implementation)
6. [Stage 4: Reward System](#stage-4-reward-system)
7. [Stage 5: Integration & Optimization](#stage-5-integration--optimization)
8. [Stage 6: Testing & Validation](#stage-6-testing--validation)
9. [Success Criteria](#success-criteria)
10. [Risk Mitigation](#risk-mitigation)

---

## Overview

### Objective
Build a Gymnasium-compatible reinforcement learning environment for scRNA-seq cluster refinement that:
- Accepts AnnData objects with preprocessed single-cell data
- Provides a 35-dimensional state representation
- Implements 5 discrete actions (split, merge, re-cluster, accept)
- Computes composite rewards balancing clustering quality and GAG enrichment
- Supports episodic RL training with PPO

### Key Design Decisions

1. **State Normalization**: Optional parameter (default: False) for comparison studies
2. **Reward Normalization**: Optional parameter (default: False) for PPO stability
3. **Resolution Bounds**: Clamp to [0.1, 2.0] with small penalty (-0.1) for boundary hits
4. **Caching**: Cache embeddings, graph structure, and gene sets; recompute metrics only when clustering changes
5. **Penalties**: Composite penalty system including degenerate states, quality degradation, and boundary violations
6. **Action Masking**: Disable invalid actions at resolution boundaries (future enhancement)

---

## Development Stages

### Stage Summary

| Stage | Component | Estimated Time | Dependencies |
|-------|-----------|---------------|--------------|
| 1 | Minimal Gymnasium Environment | 2 hours | None |
| 2 | State Representation | 6 hours | Stage 1 |
| 3 | Action Implementation | 8 hours | Stage 1, 2 |
| 4 | Reward System | 6 hours | Stage 1, 2 |
| 5 | Integration & Optimization | 8 hours | Stages 2-4 |
| 6 | Testing & Validation | Ongoing | All stages |

**Total Estimated Time**: ~30 hours + ongoing testing

---

## Stage 1: Minimal Gymnasium Environment

### Objective
Create a minimal Gymnasium-compatible environment skeleton that passes `gymnasium.utils.env_checker.check_env()`.

### Deliverables

1. **File Structure**:
   ```
   rl_sc_cluster_utils/
   └── environment/
       ├── __init__.py
       ├── clustering_env.py
       └── utils.py
   ```

2. **Core Components**:
   - `ClusteringEnv` class inheriting from `gymnasium.Env`
   - Placeholder `reset()` method returning 35-dim zero vector
   - Placeholder `step()` method with no-op actions
   - Constant reward (0.0) for all actions
   - Stub `render()` and `close()` methods

3. **Action Space**: `Discrete(5)` with action mapping:
   - 0: Split worst cluster (placeholder)
   - 1: Merge closest pair (placeholder)
   - 2: Re-cluster resolution +0.1 (placeholder)
   - 3: Re-cluster resolution -0.1 (placeholder)
   - 4: Accept (terminate episode)

4. **Observation Space**: `Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64)`

5. **Unit Tests**:
   - Test Gymnasium compliance (`check_env`)
   - Test `reset()` returns correct shape
   - Test `step()` returns correct format
   - Test action space bounds
   - Test observation space bounds

### Implementation Details

#### clustering_env.py Structure

```python
import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any
from anndata import AnnData

class ClusteringEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for scRNA-seq cluster refinement.
    
    State: 35-dimensional vector encoding clustering state
    Actions: 5 discrete actions (split, merge, re-cluster, accept)
    Reward: Composite of clustering quality and GAG enrichment
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        adata: AnnData,
        max_steps: int = 15,
        normalize_state: bool = False,
        normalize_rewards: bool = False,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Store parameters
        self.adata = adata
        self.max_steps = max_steps
        self.normalize_state = normalize_state
        self.normalize_rewards = normalize_rewards
        self.render_mode = render_mode
        
        # Action space: 5 discrete actions
        self.action_space = gym.spaces.Discrete(5)
        
        # Observation space: 35-dimensional vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(35,),
            dtype=np.float64
        )
        
        # Episode tracking
        self.current_step = 0
        self.state = None
        self.current_resolution = 0.5  # Initial Leiden resolution
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Placeholder: return zero vector
        self.state = np.zeros(35, dtype=np.float64)
        
        # Reset episode tracking
        self.current_step = 0
        self.current_resolution = 0.5
        
        info = {
            'step': self.current_step,
            'resolution': self.current_resolution,
            'n_clusters': 1  # Placeholder
        }
        
        return self.state, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Placeholder: no-op actions
        # State remains unchanged
        next_state = self.state.copy()
        
        # Placeholder: constant reward
        reward = 0.0
        
        # Check termination conditions
        terminated = (action == 4)  # Accept action
        truncated = (self.current_step >= self.max_steps - 1)
        
        # Increment step counter
        self.current_step += 1
        
        info = {
            'action': action,
            'step': self.current_step,
            'terminated': terminated,
            'truncated': truncated
        }
        
        return next_state, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment (stub for future visualization)."""
        if self.render_mode == 'human':
            # Future: UMAP visualization with cluster borders
            pass
        elif self.render_mode == 'rgb_array':
            # Future: Return RGB array for video recording
            return None
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        # Nothing to clean up yet
        pass
```

#### utils.py Structure

```python
"""Utility functions for the clustering environment."""

def validate_adata(adata: AnnData) -> None:
    """
    Validate AnnData object has required fields.
    
    Required:
    - .obsm['X_scvi']: Embedding matrix
    - .uns['neighbors']: k-NN graph
    - .obs['clusters']: Initial cluster labels (will be set at reset)
    """
    # Placeholder for Stage 1
    pass
```

### Success Criteria

- [ ] Environment passes `gymnasium.utils.env_checker.check_env()`
- [ ] All unit tests pass
- [ ] Can instantiate environment with mock AnnData
- [ ] Can run multiple episodes with random actions
- [ ] Action space and observation space correctly defined

---

## Stage 2: State Representation

### Objective
Implement the 35-dimensional state vector extraction from AnnData clustering state.

### Deliverables

1. **state.py** with:
   - `StateExtractor` class
   - Methods for each state component:
     - Global metrics (3 dims)
     - Quality metrics (3 dims)
     - GAG enrichment (28 dims)
     - Progress (1 dim)
   - Caching system for unchanged metrics
   - Normalization option

2. **State Components** (35 dimensions total):

   **Global Metrics (3)**:
   - `n_clusters / n_cells`: Normalized cluster count
   - `mean_cluster_size / n_cells`: Normalized mean cluster size
   - `cluster_size_entropy`: Entropy of cluster size distribution

   **Quality Metrics (3)**:
   - `silhouette_score`: Mean silhouette score on embeddings
   - `graph_modularity`: Newman-Girvan modularity on k-NN graph
   - `cluster_balance`: 1 - (std_size / mean_size)

   **GAG Enrichment (28 = 7 gene sets × 4 metrics)**:
   For each of 7 gene sets:
   - `mean_enrichment`: Mean AUCell score across clusters
   - `max_enrichment`: Maximum cluster enrichment
   - `F_statistic`: ANOVA F-statistic (enrichment ~ cluster)
   - `mutual_info`: Mutual information (cluster, enrichment)

   **Progress (1)**:
   - `step / max_steps`: Normalized episode progress

3. **Caching Strategy**:
   - **Permanent cache**: Embeddings, graph structure, gene sets (set at init)
   - **Episode cache**: Current clustering, resolution (reset each episode)
   - **Step cache**: Previous state values (update each step)

4. **Normalization**:
   - Optional min-max scaling per dimension
   - Preserve original values for comparison studies

### Implementation Details

#### Key Functions

```python
class StateExtractor:
    def __init__(self, adata, gene_sets, normalize=False):
        # Initialize caches
        # Store embeddings, graph, gene sets
        
    def extract_state(self, adata, step, max_steps):
        # Compute all 35 dimensions
        # Use caches where possible
        
    def _compute_global_metrics(self, adata):
        # n_clusters, mean_size, entropy
        
    def _compute_quality_metrics(self, adata):
        # silhouette, modularity, balance
        
    def _compute_gag_enrichment(self, adata, gene_sets):
        # For each gene set: mean, max, F-stat, MI
        
    def _normalize_state(self, state):
        # Optional min-max normalization
```

### Success Criteria

- [ ] State vector has exactly 35 dimensions
- [ ] All metrics computed correctly
- [ ] Caching reduces redundant computations
- [ ] Normalization works when enabled
- [ ] Handles edge cases (1 cluster, all singletons, etc.)

---

## Stage 3: Action Implementation

### Objective
Implement the 5 discrete actions: split, merge, re-cluster, and accept.

### Deliverables

1. **actions.py** with:
   - `ActionExecutor` class
   - Methods for each action type
   - Resolution clamping with penalty
   - Action validation

2. **Action Implementations**:

   **Action 0: Split Worst Cluster**
   - Identify cluster with lowest mean silhouette
   - Extract subgraph of that cluster
   - Run Leiden at `current_resolution + 0.2` on subgraph
   - Replace original cluster with new sub-clusters
   - Update `adata.obs['clusters']`

   **Action 1: Merge Closest Pair**
   - Compute cluster centroids in embedding space
   - Find pair with minimum Euclidean distance
   - Merge clusters (reassign labels)
   - Update `adata.obs['clusters']`

   **Action 2: Re-cluster Resolution +0.1**
   - Increment resolution: `resolution = min(2.0, resolution + 0.1)`
   - If clamped: apply penalty (-0.1)
   - Run Leiden on full graph with new resolution
   - Update `adata.obs['clusters']`

   **Action 3: Re-cluster Resolution -0.1**
   - Decrement resolution: `resolution = max(0.1, resolution - 0.1)`
   - If clamped: apply penalty (-0.1)
   - Run Leiden on full graph with new resolution
   - Update `adata.obs['clusters']`

   **Action 4: Accept**
   - No-op (termination handled in `step()`)

3. **Edge Case Handling**:
   - Cannot split if only 1 cluster
   - Cannot merge if only 1 cluster
   - Cannot split singleton clusters
   - Resolution bounds enforcement

### Implementation Details

#### Key Functions

```python
class ActionExecutor:
    def __init__(self, adata, min_resolution=0.1, max_resolution=2.0):
        # Store parameters
        
    def execute(self, action, current_resolution):
        # Route to appropriate action method
        
    def _split_worst_cluster(self, adata, resolution):
        # Find worst cluster, sub-cluster, update labels
        
    def _merge_closest_pair(self, adata):
        # Find closest centroids, merge, update labels
        
    def _recluster(self, adata, resolution_delta):
        # Update resolution, clamp, run Leiden
        
    def _clamp_resolution(self, resolution):
        # Clamp and return penalty flag
```

### Success Criteria

- [ ] All 5 actions execute without errors
- [ ] Clustering labels update correctly
- [ ] Resolution clamping works with penalty
- [ ] Edge cases handled gracefully
- [ ] Actions are reversible (can undo with opposite action)

---

## Stage 4: Reward System

### Objective
Implement composite reward function balancing clustering quality and GAG enrichment.

### Deliverables

1. **rewards.py** with:
   - `RewardCalculator` class
   - Methods for each reward component
   - Penalty computation
   - Optional normalization

2. **Reward Components**:

   **Q_cluster (Clustering Quality)**:
   ```python
   Q_cluster = 0.5 * silhouette + 0.3 * modularity + 0.2 * balance
   ```

   **Q_GAG (GAG Enrichment Separation)**:
   ```python
   # For each gene set:
   F_stat = f_oneway(enrichment ~ cluster)
   MI = mutual_info_score(cluster, enrichment_bins)
   Q_GAG = mean(F_stat + MI) across gene sets
   ```

   **Penalties**:
   ```python
   penalty = 0
   if n_clusters == 1:
       penalty += 5
   if n_clusters > 0.3 * n_cells:
       penalty += 5
   penalty += n_singletons * 2
   if quality_degraded:
       penalty += max(0, old_reward - new_reward)
   if resolution_clamped:
       penalty += 0.1
   ```

   **Composite Reward**:
   ```python
   reward = alpha * Q_cluster + beta * Q_GAG - delta * penalty
   # Default: alpha=0.6, beta=0.4, delta=1.0
   ```

3. **Normalization** (Optional):
   - Running mean and standard deviation
   - Update after each episode
   - Normalize: `(reward - mean) / std`

### Implementation Details

#### Key Functions

```python
class RewardCalculator:
    def __init__(
        self,
        gene_sets,
        alpha=0.6,
        beta=0.4,
        delta=1.0,
        normalize=False
    ):
        # Store parameters
        # Initialize normalization stats if needed
        
    def compute_reward(
        self,
        adata,
        previous_reward=None,
        resolution_clamped=False
    ):
        # Compute Q_cluster
        # Compute Q_GAG
        # Compute penalties
        # Combine into composite reward
        # Normalize if enabled
        
    def _compute_q_cluster(self, adata):
        # Silhouette, modularity, balance
        
    def _compute_q_gag(self, adata, gene_sets):
        # ANOVA F-stat, mutual information
        
    def _compute_penalties(
        self,
        adata,
        previous_reward,
        current_reward,
        resolution_clamped
    ):
        # Degenerate states, degradation, bounds
```

### Success Criteria

- [ ] Reward computed correctly for all states
- [ ] Penalties applied appropriately
- [ ] Normalization works when enabled
- [ ] Reward increases with better clustering
- [ ] Reward decreases with penalties

---

## Stage 5: Integration & Optimization

### Objective
Integrate all components, optimize performance, and add visualization.

### Deliverables

1. **Complete clustering_env.py**:
   - Integrate `StateExtractor`
   - Integrate `ActionExecutor`
   - Integrate `RewardCalculator`
   - Implement caching system
   - Add validation checks

2. **Performance Optimizations**:
   - Cache embeddings and graph structure
   - Only recompute metrics affected by last action
   - Vectorize GAG enrichment computation
   - Profile and optimize bottlenecks

3. **Render Method**:
   - UMAP visualization of cells
   - Color by cluster assignment
   - Overlay action history
   - Save to file or display

4. **Validation Metrics**:
   - Independent scoring function (separate from reward)
   - Final cluster quality metrics
   - Action sequence analysis

### Implementation Details

#### Integration Points

```python
class ClusteringEnv(gym.Env):
    def __init__(self, ...):
        # Initialize StateExtractor
        # Initialize ActionExecutor
        # Initialize RewardCalculator
        
    def reset(self, ...):
        # Reset clustering to initial state
        # Extract initial state vector
        # Reset caches
        
    def step(self, action):
        # Execute action (modify clustering)
        # Extract new state
        # Compute reward
        # Check termination
        # Update caches
```

### Success Criteria

- [ ] All components integrated seamlessly
- [ ] Performance meets targets (<1s per step)
- [ ] Render produces meaningful visualizations
- [ ] Validation metrics computed correctly
- [ ] Environment ready for PPO training

---

## Stage 6: Testing & Validation

### Objective
Comprehensive testing at unit, integration, and system levels.

### Deliverables

1. **Unit Tests**:
   - `test_state.py`: State extraction correctness
   - `test_actions.py`: Action execution correctness
   - `test_rewards.py`: Reward computation correctness
   - `test_utils.py`: Utility function correctness

2. **Integration Tests**:
   - `test_clustering_env.py`: Full episode execution
   - `test_caching.py`: Caching system correctness
   - `test_edge_cases.py`: Edge case handling

3. **System Tests**:
   - Run with real AnnData objects
   - Validate against known clusterings
   - Performance benchmarking
   - Memory profiling

4. **Visual Debugging**:
   - Render method output validation
   - Action sequence visualization
   - State evolution plots

### Test Coverage Goals

- Unit tests: >90% coverage
- Integration tests: All critical paths covered
- System tests: Real data validation

---

## Success Criteria

### Stage 1 (Minimal Environment)
- [ ] Passes `gymnasium.utils.env_checker.check_env()`
- [ ] All unit tests pass
- [ ] Can instantiate and run episodes

### Stage 2 (State Representation)
- [ ] 35-dim state vector computed correctly
- [ ] Caching reduces redundant computations
- [ ] Normalization works when enabled

### Stage 3 (Actions)
- [ ] All 5 actions execute correctly
- [ ] Clustering updates properly
- [ ] Edge cases handled

### Stage 4 (Rewards)
- [ ] Composite reward computed correctly
- [ ] Penalties applied appropriately
- [ ] Normalization works when enabled

### Stage 5 (Integration)
- [ ] All components work together
- [ ] Performance meets targets
- [ ] Render produces visualizations

### Stage 6 (Testing)
- [ ] >90% test coverage
- [ ] All tests pass
- [ ] Real data validation successful

---

## Risk Mitigation

### Technical Risks

1. **Performance Bottlenecks**
   - Risk: Reward computation too slow
   - Mitigation: Aggressive caching, vectorization, profiling

2. **State Space Complexity**
   - Risk: 35 dimensions may be too large/small
   - Mitigation: Start with proposal, iterate based on results

3. **Action Implementation Bugs**
   - Risk: Actions corrupt clustering state
   - Mitigation: Extensive testing, validation checks

4. **Reward Function Design**
   - Risk: Reward doesn't guide learning effectively
   - Mitigation: Start with proposal, tune with Optuna

### Data Risks

1. **Missing Required Fields**
   - Risk: AnnData missing embeddings/graph
   - Mitigation: Validation at initialization, clear error messages

2. **Gene Sets Not Provided**
   - Risk: GAG gene sets unavailable
   - Mitigation: Flexible interface, allow empty gene sets (fallback to quality only)

### Algorithm Risks

1. **PPO Compatibility**
   - Risk: Environment not compatible with PPO
   - Mitigation: Follow Gymnasium standards, test with PPO early

2. **Exploration Issues**
   - Risk: Agent gets stuck in local optima
   - Mitigation: Entropy regularization, action masking, diverse initializations

---

## Next Steps

1. **Immediate**: Implement Stage 1 (Minimal Environment)
2. **After Stage 1**: Review and iterate before Stage 2
3. **Ongoing**: Document design decisions and rationale
4. **Future**: Extend with action masking, advanced caching, etc.

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX  
**Status**: Planning Complete → Ready for Stage 1 Implementation

