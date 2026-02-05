# Contributing Guidelines

Thank you for your interest in contributing to the Root-Finding Methods Analysis project! This document provides guidelines for contributing code, documentation, and other improvements.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Types of Contributions](#types-of-contributions)
- [Review Process](#review-process)

## Code of Conduct

### Our Standards

- Be respectful and inclusive in all interactions
- Focus on what is best for the project and the community
- Show empathy towards other contributors
- Accept constructive criticism gracefully
- Prioritize educational value and code clarity

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or inflammatory comments
- Personal attacks
- Publishing others' private information

## How to Contribute

### Reporting Issues

Before creating an issue, please check if a similar one already exists.

**Bug Reports** should include:
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Python version, OS, and browser (for UI issues)
- Screenshots if applicable

**Feature Requests** should include:
- Clear description of the proposed feature
- Use case / motivation
- Suggested implementation approach (if you have one)
- Any potential challenges or trade-offs

### Suggesting Enhancements

We welcome suggestions for:
- New test functions with interesting convergence properties
- Additional numerical methods (e.g., Brent's, Ridder's)
- Improved visualizations
- Better educational content
- Performance optimizations
- UI/UX improvements

## Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Math374P2.git
   cd Math374P2
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/jakujobi/Math374P2.git
   ```

4. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed setup instructions.

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://peps.python.org/pep-0008/) with these specifics:

- **Line length**: 100 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organize as: standard library, third-party, local
- **Quotes**: Use double quotes for strings

### Type Annotations

Always use type hints:

```python
def bisection_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    delta: float = 1e-10
) -> Dict[str, Any]:
    """Function implementation."""
    pass
```

### Documentation

#### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description (one line).
    
    Extended description with more details, mathematical
    background, algorithm explanation, etc.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When and why
        
    Examples:
        >>> function_name(1, 2)
        3
    """
```

#### Comments

- Use comments sparingly, only for complex logic
- Prefer self-documenting code with clear variable names
- Explain *why*, not *what* (code should explain what)

### Naming Conventions

```python
# Functions and variables
def calculate_root():
    error_tolerance = 1e-10
    
# Classes
class NumericalMethod:
    pass
    
# Constants
MAX_ITERATIONS = 100
DEFAULT_TOLERANCE = 1e-10

# Private functions/variables
def _internal_helper():
    pass
```

### Code Organization

- Keep functions focused on a single task
- Limit function length to ~50 lines when possible
- Group related functions together in modules
- Avoid circular imports

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
Explain what changed and why, not how (code shows how).

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
- Reference issues: "Fixes #123" or "Related to #456"
```

**Good examples**:
```
Add Brent's method implementation

Implement Brent's root-finding method which combines bisection,
secant, and inverse quadratic interpolation. More robust than
secant method while maintaining similar performance.

Closes #42
```

**Bad examples**:
```
Fixed stuff
Update code
Changes
```

### Pull Request Process

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests** and verify the application works:
   ```bash
   streamlit run streamlit_app.py
   # Manually test your changes
   ```

3. **Create pull request** with:
   - Clear title describing the change
   - Detailed description of what and why
   - Screenshots for UI changes
   - Reference to related issues

4. **PR template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Tested manually
   - [ ] All numerical methods still work
   - [ ] UI displays correctly
   - [ ] No console errors
   
   ## Screenshots (if applicable)
   Attach screenshots here
   
   ## Related Issues
   Fixes #123
   ```

### Code Review Checklist

Before submitting, verify:

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Type hints are used
- [ ] No commented-out code or debug prints
- [ ] Changes are minimal and focused
- [ ] Commit messages are clear
- [ ] Application runs without errors
- [ ] UI is responsive and intuitive
- [ ] Documentation is updated if needed

## Types of Contributions

### Adding a Test Function

Example contribution:

1. **Add function to `test_functions.py`**:
   ```python
   def f4(x: float) -> float:
       """f₄(x) = tan(x) - x"""
       return np.tan(x) - x
   
   def df4(x: float) -> float:
       """f₄'(x) = sec²(x) - 1"""
       return 1 / (np.cos(x)**2) - 1
   ```

2. **Register in FUNCTIONS dict**:
   ```python
   FUNCTIONS["f4"] = {
       "function": f4,
       "derivative": df4,
       "display_name": "f₄(x) = tan(x) - x",
       "latex": r"f_4(x) = \tan(x) - x",
       "description": "Transcendental equation with multiple roots",
       "known_roots": [0.0],  # Add known roots for validation
       "suggested_intervals": [(-1, 1), (3, 4)]
   }
   ```

3. **Test thoroughly**:
   - Verify all three methods work on the new function
   - Check edge cases (discontinuities, multiple roots, etc.)
   - Document any special considerations

### Adding a Numerical Method

Example: Adding False Position method

1. **Implement in `numerical_methods.py`**:
   ```python
   def false_position_method(
       f: Callable[[float], float],
       a: float,
       b: float,
       delta: float = 1e-10,
       epsilon: float = 1e-10,
       max_iterations: int = 100
   ) -> Dict[str, Any]:
       """
       Implements the False Position (Regula Falsi) method.
       
       [Detailed docstring here]
       """
       # Implementation following standard result format
       pass
   ```

2. **Add to `streamlit_app.py`**:
   - Update `run_root_finding_methods()` to call your method
   - Add UI controls in sidebar for method parameters
   - Include in comparison visualizations

3. **Update documentation**:
   - Add method explanation to `report.py`
   - Add pseudocode
   - Update README.md features list

### Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or use cases
- Improve code comments
- Enhance README or other docs
- Add diagrams or visualizations

### Improving Visualizations

Examples:

- Better color schemes for accessibility
- Interactive plots (if Streamlit supports)
- Animation improvements
- Additional plot types (contour plots, 3D, etc.)
- Export functionality (save plots as images)

## Review Process

1. **Automatic checks**: Currently none, but code should be manually tested

2. **Maintainer review**: Project maintainer will review within 1-2 weeks

3. **Feedback**: Address any requested changes promptly

4. **Merge**: Once approved, maintainer will merge the PR

## Getting Help

### Where to Ask Questions

- Open a GitHub Issue with the "question" label
- Email the maintainer: [John Akujobi](https://github.com/jakujobi)
- Check existing documentation: [README.md](../README.md), [ARCHITECTURE.md](ARCHITECTURE.md), [DEVELOPMENT.md](DEVELOPMENT.md)

### Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Guide](https://matplotlib.org/stable/users/index.html)

## Recognition

Contributors will be recognized in:
- Git commit history
- Potential CONTRIBUTORS.md file (if project grows)
- Acknowledgments in documentation

Thank you for contributing to making numerical analysis more accessible and understandable!

---

**Questions?** Open an issue or contact [@jakujobi](https://github.com/jakujobi)
