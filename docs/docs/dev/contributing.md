# Contributing to RLscCluster

Thank you for your interest in contributing to RLscCluster! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** following our guidelines
5. **Test your changes** thoroughly
6. **Submit a pull request**

## Development Setup

See the [Setup Guide](setup.md) for detailed instructions.

Quick start:
```bash
git clone https://github.com/yourusername/rl_sc_cluster.git
cd rl_sc_cluster
make venv
source venv/bin/activate
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-state-normalization`
- `fix/reward-calculation-bug`
- `docs/update-api-reference`
- `test/add-integration-tests`
- `refactor/simplify-action-logic`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Build/tooling changes

**Example:**
```
feat(environment): add state normalization option

- Add normalize_state parameter to ClusteringEnv
- Implement min-max scaling for state vector
- Add unit tests for normalization
- Update documentation

Closes #42
```

### Code Style

We follow PEP 8 with some modifications:

- **Line length:** 99 characters (configured in pyproject.toml)
- **Formatter:** Black
- **Import sorter:** isort
- **Linter:** flake8

**Before committing:**
```bash
make format  # Auto-format code
make lint    # Check code quality
```

### Testing

All changes must include tests:

```bash
make test     # Run all tests
make test-env # Run environment tests
```

**Coverage requirement:** >90% for new code

### Documentation

Update documentation for:

- New features
- API changes
- Configuration changes
- Breaking changes

Documentation is in `docs/docs/` and uses MkDocs.

## Pull Request Process

### 1. Prepare Your PR

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### 2. Create Pull Request

**Title:** Clear, descriptive summary

**Description should include:**
- What changes were made
- Why the changes were made
- How to test the changes
- Related issues (if any)

**Template:**
```markdown
## Description
Brief description of changes

## Motivation
Why are these changes needed?

## Changes
- Change 1
- Change 2

## Testing
How to test these changes

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted
- [ ] No linting errors

Closes #issue_number
```

### 3. Review Process

- Maintainers will review your PR
- Address any feedback
- Keep PR updated with main branch
- Once approved, maintainers will merge

## Types of Contributions

### Bug Reports

**Before submitting:**
- Check if bug already reported
- Verify bug exists in latest version
- Collect relevant information

**Bug report should include:**
- Clear title
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment (OS, Python version, package versions)
- Error messages/stack traces
- Minimal code example

**Template:**
```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: macOS 13.0
- Python: 3.10.0
- Package version: 0.0.1

**Error Message**
```
Error traceback here
```

**Minimal Example**
```python
# Code to reproduce
```
```

### Feature Requests

**Before submitting:**
- Check if feature already requested
- Consider if it fits project scope
- Think about implementation

**Feature request should include:**
- Clear title
- Problem statement
- Proposed solution
- Alternatives considered
- Additional context

### Documentation Improvements

Documentation contributions are always welcome!

**Types:**
- Fix typos/grammar
- Clarify explanations
- Add examples
- Improve organization
- Add missing documentation

**Process:**
1. Edit files in `docs/docs/`
2. Test locally: `make docs-serve`
3. Submit PR

### Code Contributions

**Areas for contribution:**
- Stage 2: State representation
- Stage 3: Action implementation
- Stage 4: Reward calculation
- Stage 5: Integration & optimization
- Additional features
- Performance improvements
- Bug fixes

**Before starting:**
- Check existing issues/PRs
- Discuss major changes in an issue first
- Follow development roadmap

## Development Stages

### Current: Stage 2 (Complete)
- Gymnasium-compatible environment
- Real 35-dimensional state extraction
- StateExtractor class with full metrics
- 46 comprehensive tests passing

### Next: Stage 3
- Action implementations (split, merge, re-cluster)
- Resolution bounds handling
- Validation

See [Development Plan](../environment/development_plan.md) for details.

## Code Review Guidelines

### For Authors

- Keep PRs focused and small
- Respond to feedback promptly
- Be open to suggestions
- Update PR based on feedback

### For Reviewers

- Be constructive and respectful
- Explain reasoning for suggestions
- Approve when ready
- Request changes if needed

## Getting Help

- **Documentation:** Check [docs](https://yourusername.github.io/rl_sc_cluster)
- **Issues:** Search [existing issues](https://github.com/yourusername/rl_sc_cluster/issues)
- **Discussions:** Ask in [GitHub Discussions](https://github.com/yourusername/rl_sc_cluster/discussions)
- **Email:** Contact maintainers

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in documentation

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.

## Questions?

Don't hesitate to ask! Open an issue or discussion if you're unsure about anything.

Thank you for contributing to RLscCluster! 🎉
