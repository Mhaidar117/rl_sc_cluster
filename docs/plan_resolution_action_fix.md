# Plan: Fix Resolution Change Actions Destroying Cluster Refinement Work

## Problem Summary

The RL environment's resolution change actions (Actions 2 and 3) completely re-cluster the entire dataset using `sc.tl.leiden()`, which **destroys all incremental cluster refinement work** performed by Split and Merge actions during the episode.

---

## 1. Where the Issue Occurs

### Location: `rl_sc_cluster_utils/environment/actions.py`

**Lines 617-653** (`_increment_resolution`) and **Lines 655-691** (`_decrement_resolution`):

```python
def _increment_resolution(self, current_resolution: float) -> Dict[str, any]:
    # ...
    new_resolution = min(self.max_resolution, current_resolution + 0.1)

    # PROBLEM: This re-clusters the ENTIRE dataset from scratch
    sc.tl.leiden(
        self.adata,
        resolution=new_resolution,
        key_added="clusters",  # Overwrites all existing cluster assignments
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
```

The same pattern exists in `_decrement_resolution`.

### Evidence from Training Data

From Episode 4 (best episode with reward 8.22):
```
Step 46: Split  -> n_clusters=103
Step 47: Split  -> n_clusters=104
Step 48: Split  -> n_clusters=104
Step 49: Split  -> n_clusters=104
Step 50: Res+   -> n_clusters=79   <-- All split work destroyed!
```

The agent spent 49 steps building up to 104 refined clusters, then one `Res+` action wiped everything and produced a fresh 79-cluster Leiden partition.

---

## 2. How This Undermines PPO-Based Cluster Refinement

### 2.1 Violation of Markov Property Assumptions

PPO assumes that actions have **consistent, predictable effects** on state transitions. The current design violates this:

- **Split/Merge actions**: Incremental changes (add/remove 1-5 clusters)
- **Resolution actions**: Catastrophic state reset (can change cluster count by 20-50+)

The agent cannot learn a coherent policy when the same action type produces wildly different outcomes depending on hidden state (accumulated refinements).

### 2.2 Credit Assignment Failure

PPO uses temporal difference learning to assign credit for rewards. When resolution changes destroy previous work:

1. **Rewards from split/merge work vanish** - The carefully refined clustering that earned high Q_GAG scores is replaced
2. **Agent receives misleading signals** - A `Res+` after 40 good splits appears to "cause" a reward drop, but the real cause is the state reset
3. **Value function cannot generalize** - The value of a state depends on whether a resolution change will occur later, which the agent cannot predict

### 2.3 Exploration Becomes Destructive

High entropy coefficient (0.15) encourages action diversity, but:
- Random `Res+`/`Res-` actions during exploration destroy any discovered good states
- The agent cannot "save" promising cluster configurations
- Learning degrades to repeatedly discovering and losing good states

### 2.4 Observed Training Behavior

From the metrics:
- Initial and final UMAPs look nearly identical despite 50 steps
- Cluster counts oscillate between ~77-104 without consistent improvement
- Episode rewards don't correlate with final cluster quality (best episode ends at 79 clusters, not 104)

---

## 3. Recommended Changes

### Option A: Remove Resolution Actions Entirely (Simplest)

**Rationale**: If the goal is to refine an initial Leiden clustering using biological knowledge (GAG enrichment), global resolution changes are counterproductive.

**Changes**:
1. Remove actions 2 and 3 from action space
2. Reduce `action_space` to `Discrete(3)`: Split, Merge, Accept
3. Update state representation if resolution was encoded

**Pros**: Simple, eliminates the problem entirely
**Cons**: Loses ability to explore different base clustering granularities

### Option B: Resolution Only Affects Future Splits (Recommended)

**Rationale**: Keep resolution as a "tool setting" that affects how splits are performed, not a global re-clustering trigger.

**Changes to `actions.py`**:

```python
def _increment_resolution(self, current_resolution: float) -> Dict[str, any]:
    """
    Increment the split resolution parameter.

    This affects the resolution used for FUTURE split operations,
    but does NOT re-cluster existing clusters.
    """
    new_resolution = min(self.max_resolution, current_resolution + 0.1)
    clamped = (new_resolution == self.max_resolution
               and current_resolution < self.max_resolution)

    # Store for future split operations - DO NOT re-cluster
    # The resolution is used in _split_worst_cluster() at line 270

    return {
        "success": True,
        "error": None,
        "resolution_clamped": clamped,
        "no_change": new_resolution == current_resolution,
        "new_resolution": new_resolution,
    }
```

**Additionally update `_split_worst_cluster()`** to use `current_resolution` for sub-clustering (already does this at line 270).

**Pros**:
- Preserves incremental refinement work
- Resolution still has meaning (controls split granularity)
- Minimal code changes

**Cons**:
- Resolution changes become "delayed effect" actions
- May need to adjust reward signal for resolution actions

### Option C: Soft Resolution Adjustment (Advanced)

**Rationale**: Allow resolution to influence clustering without full reset by merging/splitting clusters to approximate the new resolution's cluster count.

**Changes**:
```python
def _increment_resolution(self, current_resolution: float) -> Dict[str, any]:
    """Increase resolution by splitting the most splittable clusters."""
    new_resolution = min(self.max_resolution, current_resolution + 0.1)

    # Estimate how many more clusters we'd have at new resolution
    # by computing Leiden on a sample or using heuristics
    target_increase = self._estimate_cluster_increase(new_resolution)

    # Perform multiple splits to approximate new resolution
    for _ in range(target_increase):
        self._split_worst_cluster(new_resolution)

    return {...}
```

**Pros**: Resolution changes feel natural, preserves most work
**Cons**: Complex to implement, may not produce consistent results

---

## 4. Implementation Plan (Option B Recommended)

### Step 1: Modify `_increment_resolution()` and `_decrement_resolution()`

Remove the `sc.tl.leiden()` calls. These methods should only update the resolution parameter and return success.

```python
def _increment_resolution(self, current_resolution: float) -> Dict[str, any]:
    new_resolution = min(self.max_resolution, current_resolution + 0.1)
    clamped = (new_resolution == self.max_resolution
               and current_resolution < self.max_resolution)

    # NO re-clustering - just update the parameter
    return {
        "success": True,
        "error": None,
        "resolution_clamped": clamped,
        "no_change": new_resolution == current_resolution,
        "new_resolution": new_resolution,
    }
```

### Step 2: Verify Split Uses Current Resolution

Confirm that `_split_worst_cluster()` already uses `current_resolution` for sub-clustering (line 270):
```python
subcluster_resolution = min(self.max_resolution, current_resolution + 0.2)
```

This is correct - splits will use a resolution relative to the current setting.

### Step 3: Update Documentation

Update docstrings to clarify:
- Resolution actions adjust the "split granularity" parameter
- They do not re-cluster existing assignments
- Higher resolution = finer splits, lower resolution = coarser splits

### Step 4: Consider Reward Implications

Resolution changes now have no immediate effect on state metrics (Q_cluster, Q_GAG). Options:
1. **No change needed** - Agent learns resolution affects future splits
2. **Add small penalty** for resolution changes to discourage spam
3. **Remove resolution from state** if it's no longer directly observable

### Step 5: Test and Validate

1. Run training with modified actions
2. Verify UMAPs show progressive refinement
3. Check that cluster counts monotonically increase (splits) or decrease (merges)
4. Confirm episode trajectories show consistent state evolution

---

## 5. Files to Modify

| File | Changes |
|------|---------|
| `rl_sc_cluster_utils/environment/actions.py` | Remove `sc.tl.leiden()` from resolution methods |
| `rl_sc_cluster_utils/environment/clustering_env.py` | Update docstrings (optional) |
| `scripts/test_ppo_minimal.py` | No changes needed |
| `tests/env_test/` | Add tests for new resolution behavior |

---

## 6. Expected Outcomes

After implementing Option B:

1. **UMAPs will show real differences** - Initial vs final clustering will reflect accumulated splits/merges
2. **Training will be more stable** - No more catastrophic state resets mid-episode
3. **Agent can learn meaningful policy** - "Split when GAG heterogeneity is high, merge when GAG profiles match"
4. **Rewards will correlate with actions** - Good split decisions lead to sustained reward improvements

---

## 7. Alternative Consideration: Remove Resolution Actions

If after implementing Option B the resolution actions prove useless (agent ignores them or they add noise), consider simplifying to a 3-action space:

- **Action 0**: Split worst cluster (GAG-aware)
- **Action 1**: Merge closest pair (GAG-aware)
- **Action 2**: Accept (terminate)

This would make the environment purely about incremental refinement using biological domain knowledge, which aligns better with the project's goal of "GAG-Sulfation-aware Refinement."
