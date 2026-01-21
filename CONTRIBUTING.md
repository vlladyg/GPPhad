# Contributing to GPPhad

Thank you for your interest in contributing to GPPhad! This document provides guidelines and information for contributors.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

---

## Getting Started

### Prerequisites

- Python 3.5+
- Git
- Understanding of Gaussian Processes and thermodynamics (helpful but not required)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/phase_diagrams-by-gaussian-process.git
cd phase_diagrams-by-gaussian-process
git remote add upstream https://github.com/ORIGINAL/phase_diagrams-by-gaussian-process.git
```

---

## Development Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install -e .
pip install pytest numpy scipy pandas gmpy2 dill matplotlib
```

### Verify Installation

```bash
python -c "import GPPhad; print('GPPhad imported successfully')"
```

---

## Making Changes

### Branch Naming

- `feature/description` — New features
- `fix/description` — Bug fixes
- `docs/description` — Documentation updates
- `refactor/description` — Code refactoring

### Workflow

```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes...

# Commit with descriptive message
git add .
git commit -m "Add: description of the feature"

# Keep branch updated
git fetch upstream
git rebase upstream/main

# Push to your fork
git push origin feature/my-new-feature
```

---

## Code Style

### General Guidelines

- Follow PEP 8 style guide
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Docstring Format

```python
def function_name(param1, param2):
    """Brief description of function.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
        
    Returns
    -------
    type
        Description of return value
        
    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    """
    pass
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `compute_var` |
| Classes | PascalCase | `GP_full` |
| Constants | UPPER_SNAKE | `BOLTZMANN_CONST` |
| Private | _leading_underscore | `_internal_method` |

### High-Precision Arithmetic

Always use GMPY2 for numerical computations:

```python
from gmpy2 import mpfr, get_context
get_context().precision = 400

# Good
x = mpfr(1.5)
y = x ** 2

# Avoid (loses precision)
x = 1.5
y = x ** 2
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_gp.py

# Run with coverage
pytest --cov=GPPhad tests/
```

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_predictions.py
import pytest
import numpy as np
from gmpy2 import mpfr, get_context
get_context().precision = 400

from GPPhad import GP_full

def test_pressure_positive():
    """Test that pressure is positive for normal volumes."""
    # Setup
    GP = create_test_gp()
    
    # Test
    T = mpfr(0.04)  # 464 K
    V = mpfr(14.0)  # Å³
    P = GP.predict_P(T, V, 'sol_fcc')[0]
    
    # Assert
    assert float(P) > 0, "Pressure should be positive"

def test_entropy_ordering():
    """Test that liquid entropy > solid entropy."""
    GP = create_test_gp()
    T = mpfr(0.04)
    
    S_liq = GP.predict_S(T, 15.0, 'liq')[0]
    S_sol = GP.predict_S(T, 14.0, 'sol_fcc')[0]
    
    assert float(S_liq) > float(S_sol)
```

---

## Documentation

### Updating Docs

Documentation is in `docs/` directory:

- `README.md` — Main project documentation
- `docs/API_REFERENCE.md` — Complete API reference
- `docs/TUTORIAL.md` — Step-by-step guide
- `docs/THEORY.md` — Mathematical background

### Building Documentation

```bash
# If using Sphinx (future)
cd docs
make html
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add mathematical formulas where appropriate (LaTeX)
- Keep tutorials practical and hands-on

---

## Submitting Changes

### Pull Request Process

1. **Update documentation** — If your change affects the API
2. **Add tests** — For new functionality
3. **Update CHANGELOG.md** — Document your changes
4. **Create Pull Request** — With clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Checklist
- [ ] Code follows project style
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your changes will be merged

---

## Reporting Issues

### Bug Reports

Include:
- Python version
- GPPhad version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/tracebacks

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered

### Issue Template

```markdown
## Description
Clear description of the issue

## Environment
- Python: 3.x
- GPPhad: 1.x
- OS: Linux/Mac/Windows

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Additional Context
Any other relevant information
```

---

## Areas for Contribution

### High Priority
- Additional example notebooks
- Performance optimization
- Test coverage improvement

### Medium Priority
- Documentation improvements
- Error message clarity
- Code refactoring

### Future Features
- Multi-component systems
- GPU support
- Web visualization

---

## Contact

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [maintainer email]

Thank you for contributing to GPPhad!
