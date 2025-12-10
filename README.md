# RLscCluster

**Reinforcement Learning for scRNA-seq Clustering with GAG-Sulfation-aware Refinement**

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)

A novel approach to single-cell RNA sequencing (scRNA-seq) cluster refinement using reinforcement learning. This project integrates domain-specific biological knowledge—specifically, glycosaminoglycan (GAG) sulfation pathway expression signatures—to guide the discovery of biologically meaningful cell subtypes.

## Overview

RLscCluster provides the first Gymnasium-compatible RL environment for scRNA-seq clustering, enabling:

- **Biology-Aware Clustering**: Balances clustering quality with GAG-sulfation pathway coherence
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
adata.obsm['X_scvi'] = np.random.randn(100, 10)  # scVI embeddings

# Define gene sets for GAG enrichment
gene_sets = {
    'CS_biosynthesis': ['gene_0', 'gene_1'],
    'HS_sulfation': ['gene_2', 'gene_3'],
}

# Create environment
env = ClusteringEnv(adata, gene_sets=gene_sets, max_steps=15)

# Use the environment
state, info = env.reset()  # Performs Leiden clustering, extracts state
state, reward, terminated, truncated, info = env.step(0)
print(f"State shape: {state.shape}")  # (35,)
print(f"Clusters: {info['n_clusters']}")
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

See [Development Plan](docs/docs/environment/development_plan.md) for full roadmap.

## Documentation

📚 **Full documentation available at:** [Documentation Site](https://yourusername.github.io/rl_sc_cluster)

- [Getting Started](docs/docs/getting-started.md)
- [Development Plan](docs/docs/environment/development_plan.md)
- [API Reference](docs/docs/api/environment.md)
- [Contributing Guide](docs/docs/dev/contributing.md)

Build and serve documentation locally:
```bash
make docs-serve  # Visit http://localhost:8000
```

## Development

### Setup

```bash
make venv          # Create virtual environment
source venv/bin/activate
```

### Testing

```bash
make test          # Run all tests
make test-env      # Run environment tests
make test-cov      # Run with coverage
```

### Code Quality

```bash
make lint          # Check code quality
make format        # Format code
```

### Documentation

```bash
make docs          # Build documentation
make docs-serve    # Serve locally
make docs-deploy   # Deploy to GitHub Pages
```

## Project Structure

```
rl_sc_cluster/
├── rl_sc_cluster_utils/      # Main package
│   ├── environment/           # RL environment
│   ├── config.py              # Configuration
│   ├── dataset.py             # Data processing
│   ├── features.py            # Feature engineering
│   ├── modeling/              # Model training/inference
│   └── plots.py               # Visualization
├── tests/                     # Test suite
│   └── env_test/              # Environment tests
├── docs/                      # Documentation (MkDocs)
│   └── docs/                  # Markdown files
├── notebooks/                 # Jupyter notebooks
├── references/                # Reference materials
├── reports/                   # Generated reports
├── models/                    # Trained models
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── interim/               # Intermediate data
│   ├── processed/             # Processed data
│   └── external/              # External data
├── venv/                      # Virtual environment
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Package configuration
├── Makefile                   # Build automation
└── README.md                  # This file
```

## Key Features

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

## Research Context

This project addresses critical gaps in scRNA-seq analysis:

1. **Lack of Biology-Aware Refinement**: Standard clustering optimizes graph modularity without incorporating domain knowledge
2. **No Principled Multi-Objective Optimization**: Balancing clustering quality and biological coherence requires systematic approach
3. **Absence of RL Environments for Genomics**: No framework exists for sequential clustering decisions in scRNA-seq

## Citation

If you use RLscCluster in your research, please cite:

```bibtex
@software{rlsccluster2025,
  title = {RLscCluster: Reinforcement Learning for scRNA-seq Clustering},
  author = {Haidar, Michael and Tyagi, Shivam},
  year = {2025},
  url = {https://github.com/mhaidar117/rl_sc_cluster}
}
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Authors

- **Michael Haidar**
- **Shivam Tyagi**

## Acknowledgments

Built using:
- [Gymnasium](https://gymnasium.farama.org/) - RL environment framework
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis
- [AnnData](https://anndata.readthedocs.io/) - Annotated data structures

## Links

- 📖 [Documentation](https://yourusername.github.io/rl_sc_cluster)
- 🐛 [Issue Tracker](https://github.com/yourusername/rl_sc_cluster/issues)
- 💬 [Discussions](https://github.com/yourusername/rl_sc_cluster/discussions)

---

**Note**: This repository is currently under active development. See [Development Plan](docs/docs/environment/development_plan.md) for current status and roadmap.
