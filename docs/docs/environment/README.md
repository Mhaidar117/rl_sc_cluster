# RL Environment Documentation

This directory contains comprehensive documentation for the RL environment implementation for scRNA-seq cluster refinement.

---

## Documentation Structure

### Core Documents

1. **[Development Plan](development_plan.md)**
   - Complete 6-stage development roadmap
   - Detailed implementation plans for each stage
   - Success criteria and risk mitigation
   - **Start here** for understanding the overall approach

2. **[Design Decisions](design_decisions.md)**
   - Rationale for key design choices
   - Alternatives considered and rejected
   - Trade-offs and compromises
   - **Reference** when making implementation decisions

3. **[State Representation](state_representation.md)**
   - Complete breakdown of 35-dimensional state vector
   - Computation methods for each component
   - Caching strategy
   - Edge case handling
   - **Essential** for implementing state extraction

4. **[Action Implementation](action_implementation.md)**
   - Detailed algorithms for all 5 actions
   - Split, merge, and re-cluster logic
   - Resolution bounds handling
   - Validation and error handling
   - **Essential** for implementing actions

5. **[Reward Calculation](reward_calculation.md)**
   - Composite reward formula
   - Q_cluster and Q_GAG computation
   - Penalty mechanisms
   - Normalization strategy
   - **Essential** for implementing rewards

---

## Quick Start

### For Implementers

1. Read **[Development Plan](development_plan.md)** to understand the stages
2. Review **[Design Decisions](design_decisions.md)** for rationale
3. Start with **Stage 1**: Minimal Gymnasium Environment
4. Reference specific documents as you implement each component:
   - State extraction → [State Representation](state_representation.md)
   - Actions → [Action Implementation](action_implementation.md)
   - Rewards → [Reward Calculation](reward_calculation.md)

### For Reviewers

1. Review **[Design Decisions](design_decisions.md)** for architectural choices
2. Check **[Development Plan](development_plan.md)** for completeness
3. Validate implementation against detailed component docs

---

## Development Status

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Minimal Environment | 🔲 Not Started | Ready to begin |
| 2. State Representation | 🔲 Not Started | Depends on Stage 1 |
| 3. Action Implementation | 🔲 Not Started | Depends on Stage 1, 2 |
| 4. Reward System | 🔲 Not Started | Depends on Stage 1, 2 |
| 5. Integration | 🔲 Not Started | Depends on Stages 2-4 |
| 6. Testing | 🔲 Not Started | Ongoing |

**Legend**: 🔲 Not Started | 🟡 In Progress | ✅ Complete

---

## Key Concepts

### State Vector (35 dimensions)
- Global metrics (3): Cluster count, size, entropy
- Quality metrics (3): Silhouette, modularity, balance
- GAG enrichment (28): 7 gene sets × 4 metrics each
- Progress (1): Episode progress

### Actions (5 discrete)
- 0: Split worst cluster
- 1: Merge closest pair
- 2: Re-cluster resolution +0.1
- 3: Re-cluster resolution -0.1
- 4: Accept (terminate)

### Reward Function
```
R = α·Q_cluster + β·Q_GAG - δ·Penalty
```
- Q_cluster: Clustering quality (silhouette, modularity, balance)
- Q_GAG: GAG enrichment separation (ANOVA F-stat, mutual info)
- Penalty: Degenerate states, degradation, bounds

---

## Related Documentation

- **[Project Proposal](../project_proposal_v2.md)**: High-level project overview
- **[Getting Started](../getting-started.md)**: Project setup and installation

---

**Last Updated**: 2025-01-XX  
**Maintainer**: RL Environment Development Team

