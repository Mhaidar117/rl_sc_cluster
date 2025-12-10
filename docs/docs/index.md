# RLscCluster

**Reinforcement Learning for scRNA-seq Clustering with GAG-Sulfation-aware Refinement**

## Overview

RLscCluster is a novel approach to single-cell RNA sequencing (scRNA-seq) cluster refinement using reinforcement learning. The project integrates domain-specific biological knowledge—specifically, glycosaminoglycan (GAG) sulfation pathway expression signatures—to guide the discovery of biologically meaningful cell subtypes.

### Key Features

- **Gymnasium-Compatible RL Environment**: First-of-its-kind environment for scRNA-seq clustering
- **Biology-Aware Rewards**: Balances clustering quality with GAG-sulfation pathway coherence
- **Adaptive Refinement**: Sequential decision-making over clustering operations (split, merge, re-cluster)
- **Interpretable Actions**: Clear mapping between RL actions and clustering operations
- **Extensible Framework**: Easily adaptable to other gene programs and cell types

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/rl_sc_cluster.git
cd rl_sc_cluster

# Create virtual environment and install dependencies
make venv
source venv/bin/activate
```

### Basic Usage

```python
from anndata import AnnData
import numpy as np
from rl_sc_cluster_utils.environment import ClusteringEnv

# Create environment with your scRNA-seq data
adata = AnnData(X=np.random.randn(100, 50))
env = ClusteringEnv(adata, max_steps=15)

# Use the environment
state, info = env.reset()
state, reward, terminated, truncated, info = env.step(0)
```

### Training with PPO

```python
from stable_baselines3 import PPO

env = ClusteringEnv(adata)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
```

## Project Status

**Current Stage:** Stage 3 Complete ✅

- ✅ Gymnasium-compatible environment
- ✅ 35-dimensional state extraction (fully implemented)
- ✅ Real-time state computation from clustering
- ✅ 5 discrete actions (fully implemented and functional)
- ✅ Comprehensive test suite (83 tests, all passing)
- ✅ Complete documentation

**Next:** Stage 4 - Reward System Implementation

See [Development Plan](environment/development_plan.md) for full roadmap.

## Documentation Structure

### Getting Started
- [Setup Guide](getting-started.md) - Installation and configuration
- [Project Proposal](project_proposal_v2.md) - Detailed project overview

### RL Environment
- [Overview](environment/README.md) - Environment introduction
- [Development Plan](environment/development_plan.md) - 6-stage roadmap
- [Design Decisions](environment/design_decisions.md) - Architecture rationale
- [State Representation](environment/state_representation.md) - 35-dim state vector
- [Action Implementation](environment/action_implementation.md) - Action algorithms
- [Reward Calculation](environment/reward_calculation.md) - Reward function

### API Reference
- [Environment](api/environment.md) - ClusteringEnv API

### Development
- [Setup](dev/setup.md) - Development environment
- [Testing](dev/testing.md) - Testing guide
- [Contributing](dev/contributing.md) - Contribution guidelines
- [Code Style](dev/code_style.md) - Style conventions

### About
- [License](about/license.md) - BSD 3-Clause License
- [Changelog](about/changelog.md) - Version history

## Key Concepts

### State Space (35 dimensions)
- **Global metrics** (3): Cluster count, size, entropy
- **Quality metrics** (3): Silhouette, modularity, balance
- **GAG enrichment** (28): 7 gene sets × 4 metrics each
- **Progress** (1): Episode progress

### Actions (5 discrete)
- **0**: Split worst cluster
- **1**: Merge closest pair
- **2**: Re-cluster resolution +0.1
- **3**: Re-cluster resolution -0.1
- **4**: Accept (terminate)

### Reward Function
```
R = α·Q_cluster + β·Q_GAG - δ·Penalty
```
- `Q_cluster`: Clustering quality (silhouette, modularity, balance)
- `Q_GAG`: GAG enrichment separation (ANOVA F-stat, mutual info)
- `Penalty`: Degenerate states, degradation, bounds

## Development Commands

```bash
# Testing
make test          # Run all tests
make test-env      # Run environment tests
make test-cov      # Run with coverage

# Code Quality
make lint          # Check code quality
make format        # Format code

# Documentation
make docs          # Build documentation
make docs-serve    # Serve locally (http://localhost:8000)

# Cleanup
make clean         # Remove compiled files
make clean-all     # Deep clean
make clean-venv    # Remove virtual environment
```

## Research Context

This project addresses critical gaps in scRNA-seq analysis:

1. **Lack of Biology-Aware Refinement**: Standard clustering optimizes graph modularity without incorporating domain knowledge
2. **No Principled Multi-Objective Optimization**: Balancing clustering quality and biological coherence requires systematic approach
3. **Absence of RL Environments for Genomics**: No framework exists for sequential clustering decisions in scRNA-seq

### Scientific Significance

- **Computational Biology**: First RL environment for scRNA-seq clustering
- **Neuroscience**: Better characterization of PNN-associated cell states
- **Machine Learning**: Novel application of RL to scientific discovery

## Citation

If you use RLscCluster in your research, please cite:

```bibtex
@software{rlsccluster2025,
  title = {RLscCluster: Reinforcement Learning for scRNA-seq Clustering},
  author = {Haidar, Michael and Tyagi, Shivam},
  year = {2025},
  url = {https://github.com/yourusername/rl_sc_cluster}
}
```

## License

BSD 3-Clause License. See [License](about/license.md) for details.

## Contact

- **Authors**: Michael Haidar & Shivam Tyagi
- **GitHub**: [rl_sc_cluster](https://github.com/yourusername/rl_sc_cluster)
- **Issues**: [Issue Tracker](https://github.com/yourusername/rl_sc_cluster/issues)

## Acknowledgments

Built using:
- [Gymnasium](https://gymnasium.farama.org/) - RL environment framework
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis
- [AnnData](https://anndata.readthedocs.io/) - Annotated data structures
