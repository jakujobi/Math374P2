# Development Guide

This guide provides detailed information for developers who want to understand, modify, or extend the Root-Finding Methods Analysis application.

## Development Environment Setup

### Prerequisites

- Python 3.11 or higher
- Git
- pip (Python package installer)
- (Optional) Docker and VS Code for DevContainer support

### Quick Setup

1. **Clone and navigate to the repository**:
   ```bash
   git clone https://github.com/jakujobi/Math374P2.git
   cd Math374P2
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python3 -c "
   from Modules.numerical_methods import bisection_method
   from Modules.test_functions import f2, df2
   result = bisection_method(f2, 0.0, 2.0)
   print(f'Test successful! Root found: {result[\"root\"]:.6f}')
   "
   ```

### Using VS Code DevContainer

The repository includes a DevContainer configuration for a consistent development environment:

1. **Install prerequisites**:
   - Docker Desktop
   - VS Code
   - Remote - Containers extension

2. **Open in container**:
   - Open the repository folder in VS Code
   - Click "Reopen in Container" when prompted
   - Wait for the container to build (first time only)

3. **Environment details**:
   - Base image: `mcr.microsoft.com/devcontainers/python:1-3.11-bullseye`
   - Auto-installs all requirements on container creation
   - Automatically starts Streamlit server on port 8501
   - Includes Pylance for Python IntelliSense

## Project Structure

```
Math374P2/
├── Modules/                          # Core application modules
│   ├── __init__.py                   # Module initialization (if present)
│   ├── numerical_methods.py         # Root-finding algorithm implementations
│   │   ├── bisection_method()       # Bisection algorithm
│   │   ├── newton_method()          # Newton's algorithm
│   │   ├── secant_method()          # Secant algorithm
│   │   └── estimate_convergence_rate()  # Convergence analysis
│   │
│   ├── test_functions.py            # Test function definitions
│   │   ├── f1, df1                  # x² - 4sin(x) and derivative
│   │   ├── f2, df2                  # x² - 1 and derivative
│   │   ├── f3, df3                  # x³ - 3x² + 3x - 1 and derivative
│   │   ├── FUNCTIONS dict           # Function metadata registry
│   │   └── get_function_details()   # Function accessor
│   │
│   ├── visualization.py             # Matplotlib visualization functions
│   │   ├── plot_function()          # Function plots with roots
│   │   ├── plot_error_convergence() # Logarithmic error plots
│   │   ├── compare_convergence_rates()  # Multi-method comparison
│   │   ├── create_iteration_table() # Tabular iteration data
│   │   └── plot_function_with_iterations_animation()  # Step-by-step viz
│   │
│   └── report.py                    # Report content and rendering
│       ├── render_introduction()
│       ├── render_methods_explanation()
│       ├── render_termination_criteria()
│       ├── render_convergence_rates()
│       ├── render_method_comparison()
│       ├── render_pseudocode()
│       └── render_code_report()
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # System design and architecture
│   ├── DEVELOPMENT.md               # This file
│   └── CONTRIBUTING.md              # Contribution guidelines
│
├── .devcontainer/                   # VS Code DevContainer config
│   └── devcontainer.json            # Container definition
│
├── .github/                         # GitHub-specific files
│   └── CODEOWNERS                   # Code ownership
│
├── Project/                         # Project materials
│   └── project2.pdf                 # Original project specification
│
├── streamlit_app.py                 # Main application entry point
├── test_streamlit.py                # Basic Streamlit test
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── LICENSE                          # Apache 2.0 license
└── README.md                        # Project overview
```

## Running Locally

### Start the Development Server

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`.

### Development Mode with Auto-Reload

Streamlit automatically reloads when you save changes to Python files. No additional configuration needed!

### Running the Test Script

```bash
python test_streamlit.py
```

This verifies basic Streamlit functionality.

## Code Organization

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `numerical_methods.py` | Core algorithms | NumPy |
| `test_functions.py` | Function definitions | NumPy |
| `visualization.py` | Plotting and tables | NumPy, Matplotlib, Pandas |
| `report.py` | Documentation content | Streamlit |
| `streamlit_app.py` | UI and orchestration | All modules |

### Dependency Graph

```
streamlit_app.py
    ├── Modules/numerical_methods.py
    │   └── numpy
    ├── Modules/test_functions.py
    │   └── numpy
    ├── Modules/visualization.py
    │   ├── numpy
    │   ├── matplotlib
    │   ├── pandas
    │   └── streamlit
    └── Modules/report.py
        └── streamlit
```

## Coding Standards

### Python Style

- **PEP 8**: Follow Python Enhancement Proposal 8 style guide
- **Type Hints**: Use type annotations for function signatures
- **Docstrings**: NumPy-style docstrings for all public functions
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Example Function Template

```python
def function_name(
    param1: Type1,
    param2: Type2,
    optional_param: Type3 = default_value
) -> ReturnType:
    """
    Brief one-line description of the function.
    
    More detailed explanation if needed. Can span multiple lines
    and include mathematical notation or algorithm details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        optional_param: Description with default value
        
    Returns:
        Description of return value and its structure
        
    Raises:
        ExceptionType: When and why this exception is raised
        
    Examples:
        >>> function_name(value1, value2)
        expected_output
    """
    # Implementation
    return result
```

### Documentation Standards

1. **Inline Comments**: Use sparingly, only for complex logic
2. **Function Docstrings**: Required for all public functions
3. **Module Docstrings**: At the top of each module file
4. **Type Hints**: Always use for function parameters and returns

## Testing

### Manual Testing

Test each numerical method individually:

```python
from Modules.numerical_methods import bisection_method, newton_method, secant_method
from Modules.test_functions import f2, df2
import numpy as np

# Test on f2(x) = x^2 - 1 (known root at x = 1)

# Bisection
result = bisection_method(f2, a=0.0, b=2.0)
assert abs(result['root'] - 1.0) < 1e-6, "Bisection failed"
print("✓ Bisection method passed")

# Newton
result = newton_method(f2, df2, x0=0.5)
assert abs(result['root'] - 1.0) < 1e-6, "Newton failed"
print("✓ Newton's method passed")

# Secant
result = secant_method(f2, a=0.0, b=2.0)
assert abs(result['root'] - 1.0) < 1e-6, "Secant failed"
print("✓ Secant method passed")
```

### Validation Checklist

When making changes, verify:

- [ ] All numerical methods still converge on test functions
- [ ] Visualizations render without errors
- [ ] UI responds to all user inputs
- [ ] Error messages are clear and helpful
- [ ] Type hints are correct
- [ ] Docstrings are updated
- [ ] No new warnings in console

### Known Test Cases

| Function | Root | Method | Expected Iterations |
|----------|------|--------|---------------------|
| f₂ (x²-1) | 1.0 | Bisection [0, 2] | ~30-35 |
| f₂ (x²-1) | 1.0 | Newton (x₀=0.5) | ~4-6 |
| f₂ (x²-1) | 1.0 | Secant [0, 2] | ~6-8 |
| f₁ (x²-4sin(x)) | ~1.933 | Newton (x₀=2) | ~5-7 |

## Common Development Tasks

### Adding a New Test Function

1. **Define the function and derivative** in `test_functions.py`:
   ```python
   def f4(x: float) -> float:
       """f₄(x) = e^x - 2"""
       return np.exp(x) - 2
   
   def df4(x: float) -> float:
       """f₄'(x) = e^x"""
       return np.exp(x)
   ```

2. **Register in FUNCTIONS dictionary**:
   ```python
   FUNCTIONS["f4"] = {
       "function": f4,
       "derivative": df4,
       "display_name": "f₄(x) = eˣ - 2",
       "latex": r"f_4(x) = e^x - 2",
       "description": "Exponential function with known root at x = ln(2)",
       "known_roots": [np.log(2)],
       "suggested_intervals": [(0, 1)]
   }
   ```

3. **Test the function**:
   ```python
   from Modules.test_functions import get_function_details
   f_details = get_function_details("f4")
   print(f_details["display_name"])
   ```

4. **Verify in UI**: Run the app and check that "f₄" appears in the dropdown

### Modifying Visualization Styles

Edit `visualization.py` to change plot appearance:

```python
# Example: Change color scheme in plot_function()
ax.plot(x_vals, y_vals, 'b-', label=f"f(x)")  # Blue line
# Change to:
ax.plot(x_vals, y_vals, 'r-', linewidth=2, label=f"f(x)")  # Red, thicker
```

### Adjusting Convergence Tolerances

Modify default tolerances in `streamlit_app.py`:

```python
# Current defaults in sidebar
delta1 = st.number_input("δ₁ (step tolerance)", value=1e-10, format="%.2e")

# Change to more relaxed tolerance:
delta1 = st.number_input("δ₁ (step tolerance)", value=1e-6, format="%.2e")
```

### Adding New Report Sections

Add content in `report.py`:

```python
def render_custom_section():
    """Render a new custom section."""
    st.markdown("""
    ## Custom Analysis Section
    
    Your content here...
    """)

# Then call in render_report():
def render_report(active_function, results):
    render_introduction()
    render_methods_explanation()
    render_custom_section()  # Add here
    # ... rest of sections
```

## Debugging Tips

### Enable Streamlit Debug Mode

```bash
streamlit run streamlit_app.py --logger.level=debug
```

### Check Method Results

Add print statements or use Streamlit's `st.write()`:

```python
result = newton_method(f, df, x0)
st.write("Debug: Result dictionary", result)
```

### Inspect Iteration History

```python
for i, iteration in enumerate(result['iterations']):
    st.write(f"Iteration {i}: {iteration}")
```

### Verify Function Values

```python
from Modules.test_functions import f1, df1
import numpy as np

x = 1.933753763
print(f"f({x}) = {f1(x)}")  # Should be close to 0
print(f"f'({x}) = {df1(x)}")
```

## Performance Optimization

### Current Performance Characteristics

- **Startup time**: ~2-3 seconds (cold start on Streamlit Cloud)
- **Method execution**: <100ms for typical convergence
- **Plot rendering**: ~200-500ms per visualization
- **UI responsiveness**: Real-time (Streamlit reactive updates)

### Optimization Opportunities

1. **Cache expensive computations**:
   ```python
   @st.cache_data
   def compute_all_methods(function_id, settings):
       # Expensive computation here
       return results
   ```

2. **Reduce plot complexity**: Lower `num_points` in `plot_function()`

3. **Lazy loading**: Only render visible tab content

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| "Small derivative" error | Newton's method near critical point | Adjust initial guess x₀ |
| "No sign change" error | Bisection interval doesn't bracket root | Change interval [a, b] |
| Plot not displaying | Matplotlib backend issue | Restart Streamlit server |
| Import error | Missing dependency | `pip install -r requirements.txt` |
| Port 8501 in use | Previous Streamlit instance running | Kill process or use `--server.port 8502` |

### Logs and Debugging

- **Streamlit logs**: Displayed in terminal where you ran `streamlit run`
- **Browser console**: F12 in browser to check for JavaScript errors
- **Python exceptions**: Streamlit displays exceptions in red in the UI

## Deployment

### Streamlit Community Cloud

The application is configured for automatic deployment:

1. Push changes to GitHub
2. Streamlit Cloud detects changes
3. Automatically rebuilds and redeploys
4. Live at https://math374p2.streamlit.app/

### Local Production Mode

Disable development features:

```bash
streamlit run streamlit_app.py --server.headless true --server.port 8501
```

### Configuration File

Create `.streamlit/config.toml` for custom settings:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
primaryColor = "#4263eb"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## Resources

### Learning Resources

- **Numerical Analysis**: Burden & Faires, "Numerical Analysis" (9th ed.)
- **Python**: [Python.org documentation](https://docs.python.org/3/)
- **Streamlit**: [Streamlit documentation](https://docs.streamlit.io/)
- **NumPy**: [NumPy user guide](https://numpy.org/doc/stable/user/)
- **Matplotlib**: [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)

### Relevant Documentation

- [Root-finding algorithms on Wikipedia](https://en.wikipedia.org/wiki/Root-finding_algorithms)
- [Convergence rate analysis](https://en.wikipedia.org/wiki/Rate_of_convergence)
- [IEEE 754 floating-point](https://en.wikipedia.org/wiki/IEEE_754) (for numerical stability)

## Getting Help

If you encounter issues:

1. Check this documentation and [ARCHITECTURE.md](ARCHITECTURE.md)
2. Review the inline code comments and docstrings
3. Search for similar issues in the repository
4. Contact the project maintainer: [John Akujobi](https://github.com/jakujobi)

---

Happy coding! 🚀
