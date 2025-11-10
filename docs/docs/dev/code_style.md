# Code Style Guide

This document outlines the code style conventions for RLscCluster.

## Python Style

We follow [PEP 8](https://pep8.org/) with some modifications.

### Line Length

**Maximum:** 99 characters

Configured in `pyproject.toml`:
```toml
[tool.black]
line-length = 99
```

### Formatting

We use **Black** for automatic code formatting:

```bash
# Format code
make format

# Check formatting
black --check rl_sc_cluster_utils
```

### Import Sorting

We use **isort** with Black-compatible profile:

```bash
# Sort imports
isort rl_sc_cluster_utils

# Check import sorting
isort --check --diff rl_sc_cluster_utils
```

Configuration in `pyproject.toml`:
```toml
[tool.isort]
profile = "black"
known_first_party = ["rl_sc_cluster_utils"]
force_sort_within_sections = true
```

### Linting

We use **flake8** for linting:

```bash
# Check code quality
make lint
```

Configuration in `setup.cfg`:
```ini
[flake8]
max-line-length = 99
ignore = E731,E266,E501,C901,W503
exclude = .git,notebooks,references,models,data
```

## Naming Conventions

### Files and Modules

- **Modules:** `lowercase_with_underscores.py`
- **Packages:** `lowercase_with_underscores/`

Examples:
- `clustering_env.py`
- `state_representation.py`
- `reward_calculation.py`

### Classes

- **Classes:** `CapitalizedWords` (PascalCase)
- **Private classes:** `_LeadingUnderscore`

Examples:
```python
class ClusteringEnv:
    pass

class StateExtractor:
    pass

class _InternalHelper:
    pass
```

### Functions and Methods

- **Functions:** `lowercase_with_underscores`
- **Private functions:** `_leading_underscore`
- **Methods:** Same as functions

Examples:
```python
def compute_reward():
    pass

def extract_state_vector():
    pass

def _internal_helper():
    pass
```

### Variables

- **Variables:** `lowercase_with_underscores`
- **Constants:** `UPPERCASE_WITH_UNDERSCORES`
- **Private variables:** `_leading_underscore`

Examples:
```python
max_steps = 15
current_resolution = 0.5
DEFAULT_ALPHA = 0.6
_cache = {}
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Tuple, Dict, Any
import numpy as np

def reset(
    self,
    seed: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reset environment."""
    pass
```

## Docstrings

### Style

We use **NumPy style** docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Short one-line summary.

    Longer description if needed. Can span multiple lines.
    Explain what the function does, not how it does it.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When param1 is negative

    Examples
    --------
    >>> function_name(42, "hello")
    True

    Notes
    -----
    Additional notes if needed.

    See Also
    --------
    related_function : Related functionality
    """
    pass
```

### Class Docstrings

```python
class ClusteringEnv:
    """
    Gymnasium-compatible RL environment for scRNA-seq cluster refinement.

    This environment provides a reinforcement learning interface for
    refining scRNA-seq clustering by balancing clustering quality and
    GAG-sulfation pathway enrichment.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell data
    max_steps : int, optional
        Maximum number of steps per episode (default: 15)
    normalize_state : bool, optional
        Whether to normalize state vector (default: False)

    Attributes
    ----------
    action_space : gymnasium.spaces.Discrete
        Discrete action space with 5 actions
    observation_space : gymnasium.spaces.Box
        Continuous observation space with 35 dimensions
    current_step : int
        Current step in the episode

    Examples
    --------
    >>> env = ClusteringEnv(adata, max_steps=15)
    >>> state, info = env.reset()
    >>> state, reward, terminated, truncated, info = env.step(0)

    See Also
    --------
    gymnasium.Env : Base environment class
    """
    pass
```

### Module Docstrings

```python
"""
Clustering environment for reinforcement learning.

This module implements a Gymnasium-compatible environment for scRNA-seq
cluster refinement using reinforcement learning.
"""
```

## Code Organization

### Imports

Order imports in three groups:

1. Standard library
2. Third-party packages
3. Local modules

```python
# Standard library
from typing import Optional, Tuple
import os

# Third-party
import numpy as np
import gymnasium as gym
from anndata import AnnData

# Local
from rl_sc_cluster_utils.config import PROJ_ROOT
from rl_sc_cluster_utils.environment.utils import validate_adata
```

### Class Structure

```python
class MyClass:
    """Class docstring."""
    
    # Class variables
    class_var = 42
    
    def __init__(self, param):
        """Initialize."""
        # Instance variables
        self.param = param
        self._private_var = None
    
    # Public methods
    def public_method(self):
        """Public method."""
        pass
    
    # Private methods
    def _private_method(self):
        """Private method."""
        pass
    
    # Special methods
    def __str__(self):
        """String representation."""
        return f"MyClass(param={self.param})"
```

## Comments

### When to Comment

- **Do:** Explain why, not what
- **Do:** Document non-obvious logic
- **Do:** Reference issues/decisions
- **Don't:** State the obvious
- **Don't:** Comment bad code (refactor instead)

### Good Comments

```python
# Use higher resolution for sub-clustering to ensure proper split
subcluster_resolution = current_resolution + 0.2

# Cache embeddings to avoid recomputation (expensive operation)
self._cached_embeddings = adata.obsm['X_scvi']

# TODO(Stage 3): Replace with actual action execution
# See: docs/environment/action_implementation.md
next_state = self.state.copy()
```

### Bad Comments

```python
# Increment i
i += 1

# Set x to 5
x = 5

# Loop through items
for item in items:
    pass
```

### TODO Comments

Format: `# TODO(context): description`

```python
# TODO(Stage 2): Implement real state extraction
# TODO(performance): Optimize this loop
# TODO(bug): Fix edge case when n_clusters == 1
# TODO(refactor): Extract this into separate function
```

## Error Handling

### Exceptions

- Use specific exception types
- Provide helpful error messages
- Include context in messages

```python
# Good
if not self.action_space.contains(action):
    raise ValueError(
        f"Invalid action: {action}. Must be in range [0, {self.action_space.n-1}]."
    )

# Bad
if not self.action_space.contains(action):
    raise Exception("Bad action")
```

### Validation

Validate inputs early:

```python
def compute_reward(adata: AnnData, alpha: float, beta: float) -> float:
    """Compute reward."""
    # Validate inputs
    if not isinstance(adata, AnnData):
        raise TypeError(f"adata must be AnnData, got {type(adata)}")
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0 <= beta <= 1:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    
    # Compute reward
    ...
```

## Testing

### Test Style

- One test per function/behavior
- Clear test names
- Arrange-Act-Assert pattern

```python
def test_reset_returns_correct_state_shape(env):
    """Test that reset returns state with correct shape."""
    # Arrange
    expected_shape = (35,)
    
    # Act
    state, info = env.reset()
    
    # Assert
    assert state.shape == expected_shape
```

### Test Naming

- `test_<function>_<scenario>_<expected_result>`

Examples:
```python
def test_step_with_valid_action_returns_correct_format():
    pass

def test_step_with_invalid_action_raises_value_error():
    pass

def test_reset_with_seed_produces_reproducible_results():
    pass
```

## Best Practices

### 1. Explicit is Better Than Implicit

```python
# Good
if terminated is True:
    pass

# Bad
if terminated:
    pass
```

### 2. Avoid Magic Numbers

```python
# Good
MAX_CLUSTERS = 100
if n_clusters > MAX_CLUSTERS:
    pass

# Bad
if n_clusters > 100:
    pass
```

### 3. Use Context Managers

```python
# Good
with open(file_path) as f:
    data = f.read()

# Bad
f = open(file_path)
data = f.read()
f.close()
```

### 4. List Comprehensions

```python
# Good (simple)
squares = [x**2 for x in range(10)]

# Good (complex - use loop)
results = []
for item in items:
    if complex_condition(item):
        result = complex_transformation(item)
        results.append(result)

# Bad (too complex)
results = [complex_transformation(item) for item in items if complex_condition(item)]
```

### 5. F-Strings

```python
# Good
message = f"Episode {episode}: reward={reward:.2f}"

# Bad
message = "Episode {}: reward={:.2f}".format(episode, reward)
message = "Episode " + str(episode) + ": reward=" + str(reward)
```

## Pre-commit Hooks

Install pre-commit hooks to automatically check style:

```bash
pre-commit install
```

This will run checks before each commit:
- Black formatting
- isort import sorting
- flake8 linting
- Trailing whitespace removal
- YAML validation

## Tools

### Black

```bash
# Format file
black file.py

# Format directory
black rl_sc_cluster_utils/

# Check without modifying
black --check rl_sc_cluster_utils/

# Show diff
black --diff rl_sc_cluster_utils/
```

### isort

```bash
# Sort imports in file
isort file.py

# Sort imports in directory
isort rl_sc_cluster_utils/

# Check without modifying
isort --check rl_sc_cluster_utils/

# Show diff
isort --diff rl_sc_cluster_utils/
```

### flake8

```bash
# Check file
flake8 file.py

# Check directory
flake8 rl_sc_cluster_utils/

# Show statistics
flake8 --statistics rl_sc_cluster_utils/
```

## References

- [PEP 8](https://pep8.org/) - Python Style Guide
- [Black](https://black.readthedocs.io/) - Code Formatter
- [isort](https://pycqa.github.io/isort/) - Import Sorter
- [flake8](https://flake8.pycqa.org/) - Linter
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/) - Docstring Style

