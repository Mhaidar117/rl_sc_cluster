# Testing Guide

This guide covers testing practices and procedures for RLscCluster.

## Test Structure

```
tests/
├── env_test/                    # Environment tests
│   ├── __init__.py
│   ├── test_clustering_env.py   # ClusteringEnv tests
│   └── README.md
└── test_data.py                 # Data processing tests
```

## Running Tests

### Using Make (Recommended)

```bash
# Run all tests
make test

# Run environment tests only
make test-env

# Run with coverage report
make test-cov
```

### Using pytest Directly

```bash
# All tests with verbose output
pytest tests -v

# Specific test file
pytest tests/env_test/test_clustering_env.py -v

# Specific test function
pytest tests/env_test/test_clustering_env.py::test_gymnasium_compliance -v

# With coverage
pytest tests --cov=rl_sc_cluster_utils --cov-report=html --cov-report=term

# Stop on first failure
pytest tests -x

# Run last failed tests
pytest tests --lf

# Run tests matching pattern
pytest tests -k "test_reset" -v
```

## Test Categories

### Unit Tests

Test individual components in isolation.

**Location:** `tests/env_test/test_clustering_env.py`

**Coverage:**
- Environment initialization
- Action space validation
- Observation space validation
- Reset functionality
- Step functionality
- Termination logic
- Truncation logic

**Example:**
```python
def test_action_space(env):
    """Test action space is correctly defined."""
    assert env.action_space.n == 5
    assert env.action_space.contains(0)
    assert env.action_space.contains(4)
    assert not env.action_space.contains(5)
```

### Integration Tests

Test component interactions.

**Example:**
```python
def test_multiple_episodes(env):
    """Test running multiple episodes."""
    for episode in range(3):
        state, info = env.reset()
        done = False
        steps = 0
        while not done and steps < env.max_steps:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps <= env.max_steps
```

### Gymnasium Compliance Tests

Verify environment follows Gymnasium API.

**Example:**
```python
def test_gymnasium_compliance(env):
    """Test that environment passes Gymnasium's check_env."""
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped, skip_render_check=True)
```

## Writing Tests

### Test Structure

```python
import pytest
from rl_sc_cluster_utils.environment import ClusteringEnv

@pytest.fixture
def mock_adata():
    """Create mock AnnData for testing."""
    # Setup
    adata = create_mock_data()
    return adata

@pytest.fixture
def env(mock_adata):
    """Create environment for testing."""
    return ClusteringEnv(adata=mock_adata)

def test_feature(env):
    """Test description."""
    # Arrange
    expected = ...

    # Act
    result = env.some_method()

    # Assert
    assert result == expected
```

### Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`
- Fixtures: descriptive names (e.g., `mock_adata`, `env`)

### Assertions

Use descriptive assertions:

```python
# Good
assert env.action_space.n == 5, "Action space should have 5 actions"

# Better
import numpy as np
np.testing.assert_array_equal(state, expected_state)
```

### Parametrized Tests

Test multiple inputs:

```python
@pytest.mark.parametrize("action,expected_terminated", [
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (4, True),
])
def test_termination(env, action, expected_terminated):
    """Test termination for each action."""
    env.reset()
    _, _, terminated, _, _ = env.step(action)
    assert terminated == expected_terminated
```

### Testing Exceptions

```python
def test_invalid_action(env):
    """Test that invalid actions raise ValueError."""
    env.reset()
    with pytest.raises(ValueError):
        env.step(5)  # Out of bounds
```

## Test Coverage

### Viewing Coverage

After running `make test-cov`:

```bash
# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Goals

- **Overall:** >90%
- **Critical paths:** 100%
- **Edge cases:** Covered

### Checking Coverage

```bash
# Generate coverage report
pytest tests --cov=rl_sc_cluster_utils --cov-report=term

# Show missing lines
pytest tests --cov=rl_sc_cluster_utils --cov-report=term-missing

# Fail if coverage below threshold
pytest tests --cov=rl_sc_cluster_utils --cov-fail-under=90
```

## Continuous Integration

### GitHub Actions (Future)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests --cov=rl_sc_cluster_utils
```

## Test Fixtures

### Common Fixtures

```python
@pytest.fixture
def mock_adata():
    """Create minimal mock AnnData."""
    import numpy as np
    from anndata import AnnData

    n_obs = 100
    n_vars = 50
    X = np.random.randn(n_obs, n_vars)
    return AnnData(X=X)

@pytest.fixture
def env(mock_adata):
    """Create ClusteringEnv instance."""
    return ClusteringEnv(adata=mock_adata, max_steps=15)

@pytest.fixture
def env_with_normalization(mock_adata):
    """Create environment with normalization enabled."""
    return ClusteringEnv(
        adata=mock_adata,
        normalize_state=True,
        normalize_rewards=True
    )
```

### Fixture Scopes

```python
# Function scope (default) - run for each test
@pytest.fixture
def env(mock_adata):
    return ClusteringEnv(adata=mock_adata)

# Module scope - run once per module
@pytest.fixture(scope="module")
def expensive_data():
    return load_large_dataset()

# Session scope - run once per session
@pytest.fixture(scope="session")
def database():
    db = setup_database()
    yield db
    teardown_database(db)
```

## Mocking

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked dependency."""
    with patch('module.expensive_function') as mock_func:
        mock_func.return_value = 42
        result = function_that_calls_expensive_function()
        assert result == 42
        mock_func.assert_called_once()
```

## Test Markers

### Custom Markers

```python
# In pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]

# In test file
@pytest.mark.slow
def test_expensive_operation():
    """This test takes a long time."""
    pass

# Run tests excluding slow ones
pytest -m "not slow"
```

## Debugging Tests

### Print Debugging

```python
def test_with_debug(env):
    """Test with debug output."""
    state, info = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Info: {info}")
    assert state.shape == (35,)
```

Run with: `pytest tests -s` (shows print statements)

### Using pdb

```python
def test_with_breakpoint(env):
    """Test with debugger."""
    state, info = env.reset()
    import pdb; pdb.set_trace()  # Breakpoint
    assert state.shape == (35,)
```

Run with: `pytest tests --pdb` (drops into debugger on failure)

### Verbose Output

```bash
# Show test names and results
pytest tests -v

# Show even more detail
pytest tests -vv

# Show local variables on failure
pytest tests -l
```

## Best Practices

### 1. Test Independence

Each test should be independent:

```python
# Good - uses fixture
def test_reset(env):
    state, info = env.reset()
    assert state.shape == (35,)

# Bad - depends on previous test
def test_step():  # Missing env fixture
    state, reward, _, _, _ = env.step(0)  # env undefined
```

### 2. Clear Test Names

```python
# Good
def test_reset_returns_correct_state_shape():
    pass

# Bad
def test_1():
    pass
```

### 3. One Assertion Per Test

```python
# Good
def test_action_space_size(env):
    assert env.action_space.n == 5

def test_action_space_contains_valid_actions(env):
    assert env.action_space.contains(0)
    assert env.action_space.contains(4)

# Acceptable for related assertions
def test_observation_space(env):
    assert env.observation_space.shape == (35,)
    assert env.observation_space.dtype == np.float64
```

### 4. Test Edge Cases

```python
def test_max_steps_truncation(env):
    """Test episode truncates at max_steps."""
    env.reset()
    for i in range(env.max_steps):
        _, _, terminated, truncated, _ = env.step(0)
        if i < env.max_steps - 1:
            assert truncated is False
        else:
            assert truncated is True
```

### 5. Use Fixtures for Setup

```python
# Good
@pytest.fixture
def configured_env(mock_adata):
    return ClusteringEnv(adata=mock_adata, max_steps=10)

def test_with_fixture(configured_env):
    assert configured_env.max_steps == 10

# Bad
def test_without_fixture():
    adata = AnnData(...)  # Setup in every test
    env = ClusteringEnv(adata=adata)
    assert env.max_steps == 15
```

## Troubleshooting

### Tests Not Found

```bash
# Ensure test files start with test_
# Ensure test functions start with test_
# Check PYTHONPATH
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

### Import Errors

```bash
# Install package in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

### Fixture Not Found

```bash
# Ensure fixture is defined in same file or conftest.py
# Check fixture scope
# Verify fixture name matches parameter name
```

## Next Steps

- Review [Code Style Guide](code_style.md)
- Check [API Reference](../api/environment.md)
- Read [Contributing Guide](contributing.md)
