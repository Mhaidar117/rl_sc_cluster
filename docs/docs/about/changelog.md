# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Performance Optimizations and Visualizations
- **ClusteringCache class**: Hash-based LRU cache for clustering-dependent metrics
  - Caches Q_cluster (silhouette, modularity, balance) computations
  - Caches Q_GAG (F-statistics) computations
  - Caches state extraction metrics (quality and GAG enrichment)
  - LRU eviction with configurable max size (default: 100 entries)
  - Cache statistics tracking (hits, misses, hit rate)
- **Precomputed enrichment scores**: Static gene set enrichment scores computed once at initialization
  - Eliminates redundant enrichment score computations (6×O(n) operations per step)
  - ~30-40% reduction in GAG computation time
- **Training convergence plots**: Visualize learning progress during PPO training
  - Episode rewards over timesteps with moving average
  - Episode lengths over timesteps
  - Reward distribution histogram
  - Learning progress comparison (first vs second half)
- **Evaluation convergence plots**: Track evaluation metrics across episodes
  - Episode rewards with mean and ±1 std bands
  - Q_GAG progression over episodes
  - Q_cluster progression over episodes
  - Final cluster counts over episodes
- **UMAP visualizations**: Best episode clustering visualization
  - Initial clustering state UMAP plot
  - Final clustering state UMAP plot
  - Side-by-side comparison for best performing episode
- **Progress bars**: tqdm integration for episode tracking
  - Real-time episode progress with reward, clusters, and length info

### Changed
- **Performance**: Expected 1.5-3x speedup from caching optimizations
- **Default parameter**: `min_steps_before_accept` default changed from 20 to 10 (matches `max_steps=15`)

### Fixed
- **Bug**: `min_steps_before_accept` default (20) was greater than `max_steps` default (15), causing early termination penalty to always apply
- **Bug**: Cache invalidation on environment reset to prevent stale entries

### Stage 4 (Planned)
- Reward calculation
- Composite reward function
- Penalty mechanisms

### Stage 5 (Planned)
- Integration and optimization
- Performance improvements
- Render method implementation

### Stage 6 (Planned)
- Comprehensive testing
- Documentation completion
- Production readiness

## [0.0.3] - 2025-12-02

### Added - Stage 3 Complete
- Action execution implementation (`ActionExecutor` class)
- Action 0: Split worst cluster (by silhouette score)
- Action 1: Merge closest pair (by centroid distance)
- Action 2: Re-cluster resolution +0.1 (with clamping)
- Action 3: Re-cluster resolution -0.1 (with clamping)
- Action 4: Accept (no-op, handled in step())
- Action validation with semantic error handling
- Resolution clamping with penalty flag (for Stage 4)
- Cluster ID conversion utilities (numeric format)
- Edge case handling (singletons, single cluster, bounds)
- Comprehensive action tests (26 tests)
- Updated environment integration tests (57 tests)
- Total: 83 tests passing (up from 46)

### Changed
- `ClusteringEnv.step()` now executes real actions (replaces placeholders)
- Resolution tracking updated after re-cluster actions
- Info dict now includes `action_success`, `action_error`, `resolution_clamped`, `no_change`
- Cluster IDs automatically converted to numeric after each action

### Fixed
- Neighbors graph access pattern (uses `obsp['connectivities']` with fallback)
- Pandas categorical cluster ID handling
- Cluster mask indexing for sparse matrices
- Single cluster edge case in split action

## [0.0.2] - 2025-11-10

### Added - Stage 2 Complete
- State extraction implementation (`StateExtractor` class)
- Real 35-dimensional state vector computation
- Global metrics (3 dims): cluster count, mean size, entropy
- Quality metrics (3 dims): silhouette, modularity, balance
- GAG enrichment (28 dims): 7 gene sets × 4 metrics each
- Progress tracking (1 dim): normalized episode progress
- Optional state normalization
- Caching system for embeddings and graph structure
- Automatic neighbors graph computation
- Future-proof Leiden clustering (igraph flavor)
- Comprehensive state extraction tests (24 tests)
- Updated environment tests (22 tests)
- Total: 46 tests passing

### Dependencies Added
- scikit-learn>=1.3.0 (silhouette score, mutual information)
- scanpy>=1.9.0 (Leiden clustering, modularity)
- scipy>=1.11.0 (ANOVA F-statistic, entropy)
- igraph>=0.10.0 (graph operations)
- leidenalg>=0.9.0 (Leiden clustering)

### Changed
- State vector now computed from real clustering metrics (no longer zeros)
- Environment performs initial Leiden clustering on reset
- Added `gene_sets` parameter to `ClusteringEnv`
- Updated Leiden clustering to use igraph flavor for future compatibility

### Fixed
- Eliminated 19 Leiden backend deprecation warnings
- Fixed observation space warnings (2 remain, expected)

## [0.0.1] - 2025-01-10

### Added - Stage 1 Complete
- Initial project structure
- Minimal Gymnasium-compatible environment (`ClusteringEnv`)
- 35-dimensional observation space (placeholder)
- 5 discrete actions (placeholder)
- Episodic learning with termination/truncation
- Comprehensive test suite (20 tests, 100% pass rate)
- Virtual environment setup
- Requirements.txt with all dependencies
- Makefile with development targets
- Documentation structure with MkDocs
- Environment documentation (6 detailed guides)
- API reference documentation
- Development guides (setup, testing, contributing, code style)
- Setup guide (SETUP.md)
- Stage 1 completion summary

### Environment Features
- `ClusteringEnv` class with Gymnasium API compliance
- Placeholder state vector (zeros)
- Placeholder actions (no-op)
- Placeholder rewards (0.0)
- Proper episode management
- Optional state/reward normalization (parameters)
- Configurable max_steps

### Testing
- 20 comprehensive unit tests
- Gymnasium compliance verification
- Action/observation space validation
- Reset/step functionality tests
- Termination/truncation logic tests
- Edge case handling
- Test coverage reporting

### Documentation
- Project proposal (detailed)
- Development plan (6 stages)
- Design decisions
- State representation details
- Action implementation guide
- Reward calculation guide
- API reference
- Setup guide
- Testing guide
- Contributing guide
- Code style guide

### Infrastructure
- Virtual environment setup
- Requirements.txt
- Makefile with targets:
  - `make venv` - Create virtual environment
  - `make test` - Run all tests
  - `make test-env` - Run environment tests
  - `make test-cov` - Run with coverage
  - `make lint` - Check code quality
  - `make format` - Format code
  - `make docs` - Build documentation
  - `make docs-serve` - Serve documentation
  - `make clean` - Remove compiled files
  - `make clean-all` - Deep clean
  - `make clean-venv` - Remove virtual environment
- Pre-commit hooks configuration
- MkDocs configuration with Material theme

### Dependencies
- gymnasium>=0.29.0
- numpy>=1.24.0
- anndata>=0.9.0
- black, flake8, isort (code quality)
- pytest, pytest-cov (testing)
- mkdocs, mkdocs-material (documentation)
- python-dotenv, tqdm, typer, loguru (utilities)

## [0.0.0] - 2025-01-09

### Added
- Initial repository setup
- Project structure (Cookiecutter Data Science template)
- Basic configuration files
- License (BSD 3-Clause)
- README.md

---

## Version History

- **0.0.2** (2025-11-10): Stage 2 Complete - State Representation
- **0.0.1** (2025-01-10): Stage 1 Complete - Minimal Gymnasium Environment
- **0.0.0** (2025-01-09): Initial Setup

## Upcoming Releases

### v0.1.0 (Planned)
- Stage 3: Action Implementation
- Stage 4: Reward Calculation

### v0.2.0 (Planned)
- Stage 5: Integration & Optimization
- Stage 6: Testing & Validation
- Production-ready environment

### v1.0.0 (Planned)
- Complete RL environment
- PPO training integration
- Comprehensive documentation
- Tutorial notebooks
- Performance benchmarks

## Links

- [GitHub Repository](https://github.com/yourusername/rl_sc_cluster)
- [Documentation](https://yourusername.github.io/rl_sc_cluster)
- [Issue Tracker](https://github.com/yourusername/rl_sc_cluster/issues)
