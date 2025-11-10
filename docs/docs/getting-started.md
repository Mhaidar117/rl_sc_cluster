# Getting Started

This guide will help you get started with RLscCluster.

## Prerequisites

- Python 3.10 or higher
- Git
- Make (optional, but recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rl_sc_cluster.git
cd rl_sc_cluster
```

### 2. Set Up Virtual Environment

Using Make (recommended):
```bash
make venv
```

Or manually:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Activate Virtual Environment

```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

## Quick Start

### Basic Example

```python
from anndata import AnnData
import numpy as np
from rl_sc_cluster_utils.environment import ClusteringEnv

# Create mock data (replace with your real data)
n_cells = 100
n_genes = 50
X = np.random.randn(n_cells, n_genes)
adata = AnnData(X=X)

# Create environment
env = ClusteringEnv(adata, max_steps=15)

# Reset environment
state, info = env.reset()
print(f"Initial state shape: {state.shape}")
print(f"Info: {info}")

# Take an action
action = 0  # Split worst cluster
state, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
```

### Running Tests

```bash
# Run all tests
make test

# Run environment tests only
make test-env

# Run with coverage
make test-cov
```

### Viewing Documentation

```bash
# Build documentation
make docs

# Serve documentation locally (http://localhost:8000)
make docs-serve
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
├── docs/                      # Documentation
├── venv/                      # Virtual environment
├── requirements.txt           # Dependencies
├── Makefile                   # Build automation
└── README.md                  # Project overview
```

## Development Workflow

### 1. Make Changes

Edit files in `rl_sc_cluster_utils/` or `tests/`.

### 2. Format Code

```bash
make format
```

### 3. Check Code Quality

```bash
make lint
```

### 4. Run Tests

```bash
make test
```

### 5. Update Documentation

Edit files in `docs/docs/` and rebuild:

```bash
make docs-serve
```

## Next Steps

- Read the [Development Plan](environment/development_plan.md)
- Check the [API Reference](api/environment.md)
- Review [Design Decisions](environment/design_decisions.md)
- See [Contributing Guide](dev/contributing.md)

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Add project root to PYTHONPATH
export PYTHONPATH=/path/to/rl_sc_cluster:$PYTHONPATH

# Or install in development mode
pip install -e .
```

### Virtual Environment Issues

If virtual environment is corrupted:

```bash
make clean-venv
make venv
source venv/bin/activate
```

### Test Failures

If tests fail:

1. Ensure virtual environment is activated
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.10+)
4. Run with verbose output: `pytest -v`

## Getting Help

- Check [Documentation](https://yourusername.github.io/rl_sc_cluster)
- Open an [Issue](https://github.com/yourusername/rl_sc_cluster/issues)
- Read [Setup Guide](dev/setup.md)
- Review [Testing Guide](dev/testing.md)

## What's Next?

Now that you have RLscCluster set up, you can:

1. **Explore the Environment**: Read [Environment API](api/environment.md)
2. **Understand the Design**: Review [Design Decisions](environment/design_decisions.md)
3. **Contribute**: See [Contributing Guide](dev/contributing.md)
4. **Train Models**: Check [Environment API](api/environment.md) for current functionality

Happy clustering! 🎉
