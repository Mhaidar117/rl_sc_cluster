# Development Setup Guide

This guide will help you set up your development environment for contributing to RLscCluster.

## Prerequisites

- Python 3.10 or higher
- Git
- Make (optional, but recommended)

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rl_sc_cluster.git
cd rl_sc_cluster
```

### 2. Create Virtual Environment

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

## Development Workflow

### Running Tests

```bash
# All tests
make test

# Environment tests only
make test-env

# With coverage report
make test-cov
```

### Code Quality

```bash
# Check code style
make lint

# Auto-format code
make format
```

### Documentation

```bash
# Build documentation
make docs

# Serve documentation locally (http://localhost:8000)
make docs-serve

# Deploy to GitHub Pages
make docs-deploy
```

### Cleanup

```bash
# Remove compiled Python files
make clean

# Deep clean (includes build artifacts)
make clean-all

# Remove virtual environment
make clean-venv
```

## Project Structure

```
rl_sc_cluster/
├── rl_sc_cluster_utils/      # Main package
│   ├── environment/           # RL environment
│   │   ├── clustering_env.py  # Main environment class
│   │   ├── utils.py           # Utility functions
│   │   └── __init__.py
│   ├── config.py              # Configuration
│   ├── dataset.py             # Data processing
│   ├── features.py            # Feature engineering
│   ├── modeling/              # Model training/inference
│   │   ├── train.py
│   │   └── predict.py
│   └── plots.py               # Visualization
│
├── tests/                     # Test suite
│   ├── env_test/              # Environment tests
│   │   └── test_clustering_env.py
│   └── test_data.py
│
├── docs/                      # Documentation
│   ├── docs/                  # Markdown files
│   │   ├── environment/       # Environment docs
│   │   ├── api/               # API reference
│   │   ├── dev/               # Development guides
│   │   └── about/             # About/meta
│   └── mkdocs.yml             # MkDocs configuration
│
├── venv/                      # Virtual environment (not in git)
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Package configuration
├── Makefile                  # Build automation
├── SETUP.md                  # Setup guide
└── README.md                 # Project overview
```

## Installing Dependencies

### Core Dependencies

```bash
pip install gymnasium>=0.29.0 numpy>=1.24.0 anndata>=0.9.0
```

### Development Dependencies

```bash
pip install black flake8 isort pytest pytest-cov
```

### Documentation Dependencies

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

### All Dependencies

```bash
pip install -r requirements.txt
```

## Development Tools

### Code Formatting

We use:
- **Black** for code formatting (99 char line length)
- **isort** for import sorting
- **flake8** for linting

Configuration in `pyproject.toml` and `setup.cfg`.

### Testing

We use **pytest** for testing:

```bash
# Run specific test file
pytest tests/env_test/test_clustering_env.py -v

# Run specific test
pytest tests/env_test/test_clustering_env.py::test_gymnasium_compliance -v

# Run with markers
pytest -m "not slow" -v

# Run with coverage
pytest --cov=rl_sc_cluster_utils --cov-report=html
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pre-commit install
```

This will run checks before each commit:
- Black formatting
- isort import sorting
- flake8 linting
- Trailing whitespace removal
- YAML validation

## IDE Setup

### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Black Formatter
- isort
- Jupyter

Recommended settings (`.vscode/settings.json`):
```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true
}
```

### PyCharm

1. Set interpreter to `venv/bin/python`
2. Enable Black formatter: Preferences → Tools → Black
3. Enable pytest: Preferences → Tools → Python Integrated Tools → Testing
4. Set line length to 99: Preferences → Editor → Code Style → Python

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

### Python Version Issues

The project requires Python 3.10+. Check your version:

```bash
python --version
```

If you need a different version, use pyenv or conda:

```bash
# Using pyenv
pyenv install 3.10.0
pyenv local 3.10.0

# Using conda
conda create -n rl_sc_cluster python=3.10
conda activate rl_sc_cluster
```

### Test Failures

If tests fail:

1. Ensure virtual environment is activated
2. Ensure all dependencies are installed
3. Check Python version (3.10+)
4. Run with verbose output: `pytest -v`
5. Check specific test: `pytest path/to/test.py::test_name -v`

## Making Changes

### Workflow

1. Create a new branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `make test`
4. Format code: `make format`
5. Check linting: `make lint`
6. Commit changes: `git commit -m "Description"`
7. Push branch: `git push origin feature/your-feature`
8. Create pull request

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tooling changes

Example:
```
feat: add state normalization option to ClusteringEnv

- Add normalize_state parameter to __init__
- Implement min-max scaling for state vector
- Add tests for normalization
```

## Getting Help

- Check [documentation](https://yourusername.github.io/rl_sc_cluster)
- Open an [issue](https://github.com/yourusername/rl_sc_cluster/issues)
- Join discussions on [GitHub Discussions](https://github.com/yourusername/rl_sc_cluster/discussions)

## Next Steps

- Read [Contributing Guide](contributing.md)
- Review [Code Style Guide](code_style.md)
- Check [Testing Guide](testing.md)
- Explore [API Reference](../api/environment.md)
