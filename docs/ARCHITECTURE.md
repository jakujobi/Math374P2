# Architecture Documentation

## System Overview

The Root-Finding Methods Analysis application is a web-based educational tool built with Streamlit. It demonstrates three classical numerical methods for finding roots of nonlinear equations through interactive visualizations and comprehensive mathematical analysis.

### High-Level Architecture

The application follows a modular, layered architecture:

```
┌─────────────────────────────────────────────────────┐
│         Streamlit Web Interface Layer               │
│  (streamlit_app.py - User interaction & display)    │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Visualization│ │   Reports    │ │Test Functions│
│   Module     │ │   Module     │ │   Module     │
└──────────────┘ └──────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
              ┌──────────────────┐
              │ Numerical Methods│
              │     Module       │
              └──────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Modularity**: Components can be tested and used independently
3. **Educational Focus**: Code is written for clarity and learning, not just efficiency
4. **Robustness**: Extensive error handling for numerical edge cases
5. **Extensibility**: Easy to add new test functions or numerical methods

## Module Descriptions

### 1. streamlit_app.py (Main Application)

**Purpose**: Entry point that orchestrates the entire application

**Key Components**:
- Page configuration and custom CSS styling
- Sidebar for user input (function selection, parameter configuration)
- Tabbed interface for different views (Analysis, Report, Pseudocode, Code Report)
- Main analysis workflow that calls numerical methods and displays results

**Data Flow**:
```
User Input (sidebar) → run_root_finding_methods() → Numerical Methods
                                                   ↓
                                            Results Dictionary
                                                   ↓
                                    Visualization + Display (tabs)
```

**Key Functions**:
- `run_root_finding_methods(function_id, settings)`: Executes all three methods
- `main()`: Sets up UI and handles user interactions

### 2. Modules/numerical_methods.py

**Purpose**: Core implementations of root-finding algorithms

**Algorithms Implemented**:

#### Bisection Method
- **Convergence Order**: Linear (p = 1)
- **Requirements**: Continuous function, sign change in initial interval
- **Advantages**: Guaranteed convergence, robust
- **Disadvantages**: Slow convergence

**Key Functions**:
```python
def bisection_method(f, a, b, delta, epsilon, max_iterations) -> Dict
```

**Algorithm Steps**:
1. Verify sign change: f(a) × f(b) < 0
2. Compute midpoint: c = (a + b) / 2
3. Evaluate f(c)
4. Check convergence: |b - a| < δ₁ or |f(c)| < δ₂
5. Select subinterval with sign change
6. Repeat until convergence or max iterations

#### Newton's Method (Newton-Raphson)
- **Convergence Order**: Quadratic (p = 2) for simple roots
- **Requirements**: Differentiable function, good initial guess
- **Advantages**: Very fast convergence when it works
- **Disadvantages**: Can fail if f'(x) ≈ 0 or poor initial guess

**Key Functions**:
```python
def newton_method(f, df, x0, delta1, delta2, epsilon, max_iterations) -> Dict
```

**Algorithm Steps**:
1. Compute f(xₙ) and f'(xₙ)
2. Check if |f'(xₙ)| < ε (derivative too small)
3. Update: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
4. Check convergence: |xₙ₊₁ - xₙ| < δ₁ or |f(xₙ₊₁)| < δ₂
5. Repeat until convergence or max iterations

#### Secant Method
- **Convergence Order**: Superlinear (p ≈ 1.618, golden ratio)
- **Requirements**: Two initial guesses, no derivative needed
- **Advantages**: No derivative calculation, faster than bisection
- **Disadvantages**: Not guaranteed to converge, slower than Newton

**Key Functions**:
```python
def secant_method(f, a, b, delta1, delta2, max_iterations) -> Dict
```

**Algorithm Steps**:
1. Start with two points: x₀ and x₁
2. Approximate derivative: (f(xₙ) - f(xₙ₋₁))/(xₙ - xₙ₋₁)
3. Update: xₙ₊₁ = xₙ - f(xₙ) × (xₙ - xₙ₋₁)/(f(xₙ) - f(xₙ₋₁))
4. Check convergence: |xₙ₊₁ - xₙ| < δ₁ or |f(xₙ₊₁)| < δ₂
5. Repeat until convergence or max iterations

**Convergence Rate Estimation**:
```python
def estimate_convergence_rate(error_history) -> float
```

Uses log-log linear regression to estimate convergence order:
- log|eₙ₊₁| ≈ p × log|eₙ| + log(C)
- Slope of regression line = convergence rate p

### 3. Modules/test_functions.py

**Purpose**: Provides test functions and their derivatives

**Test Functions**:

| Function | Equation | Derivative | Known Roots | Characteristics |
|----------|----------|------------|-------------|-----------------|
| f₁ | x² - 4sin(x) | 2x - 4cos(x) | x ≈ 0, ±1.93 | Mixed polynomial/trig |
| f₂ | x² - 1 | 2x | x = ±1 | Simple quadratic |
| f₃ | x³ - 3x² + 3x - 1 | 3x² - 6x + 3 | x = 1 (triple) | Multiple root |

**Data Structure**:
```python
FUNCTIONS = {
    "f1": {
        "function": callable,
        "derivative": callable,
        "display_name": str,
        "latex": str,
        "description": str,
        "known_roots": List[float],
        "suggested_intervals": List[Tuple[float, float]]
    }
}
```

**Key Functions**:
- `get_function_details(function_id)`: Returns complete function metadata
- `get_all_functions()`: Returns dictionary of all available functions

### 4. Modules/visualization.py

**Purpose**: Creates publication-quality visualizations using Matplotlib

**Visualization Components**:

#### Function Plots
```python
plot_function(f, x_range, root, points) -> Figure
```
- Plots the function over a specified range
- Marks roots with red circles
- Shows iteration points with color gradient (viridis colormap)
- Includes x-axis reference line

#### Error Convergence Plots
```python
plot_error_convergence(error_history, method_name, rate) -> Figure
```
- Logarithmic y-axis to show exponential convergence
- Displays estimated convergence rate
- Line plot with markers for each iteration

#### Comparative Analysis
```python
compare_convergence_rates(results) -> Figure
```
- Overlays error histories from all three methods
- Logarithmic scale for easy comparison
- Color-coded lines with legend

#### Iteration Tables
```python
create_iteration_table(iterations, method) -> DataFrame
```
- Formats iteration data into readable tables
- Method-specific columns (e.g., midpoint for bisection, derivative for Newton)
- Scientific notation for numerical values

#### Iteration Animations
```python
plot_function_with_iterations_animation(f, iterations, method, x_range) -> List[Figure]
```
- Creates frame-by-frame visualization of iteration process
- Shows tangent lines (Newton), secant lines (Secant), or intervals (Bisection)
- Indicates next guess with visual markers

### 5. Modules/report.py

**Purpose**: Provides detailed mathematical explanations and documentation

**Report Sections**:
- Introduction to root-finding methods
- Detailed algorithm explanations
- Termination criteria discussion
- Convergence rate theory
- Method comparison and trade-offs
- Function-specific analysis
- Pseudocode for each method
- Implementation details and code structure

**Key Functions**:
- `render_introduction()`: Overview and context
- `render_methods_explanation()`: Algorithm theory
- `render_termination_criteria()`: Stopping conditions
- `render_convergence_rates()`: Theoretical analysis
- `render_method_comparison(results)`: Comparative analysis
- `render_pseudocode()`: Algorithm pseudocode
- `render_code_report()`: Technical documentation

## Data Flow

### Complete Workflow

```
User → Streamlit UI → streamlit_app.py
                            ↓
                    Select function from test_functions.py
                            ↓
                    Run numerical_methods.py (all 3 methods)
                            ↓
                    Collect results (roots, iterations, errors)
                            ↓
            ┌───────────────┴───────────────┐
            ▼                               ▼
    visualization.py                   report.py
    (plots & tables)                   (documentation)
            │                               │
            └───────────────┬───────────────┘
                            ▼
                    Display in Streamlit UI
                            ▼
                          User
```

### Result Dictionary Structure

Each numerical method returns a standardized dictionary:

```python
{
    'root': float,                    # Approximate root
    'iterations': List[Dict],         # Step-by-step details
    'converged': bool,                # Success flag
    'iterations_count': int,          # Total iterations
    'error_history': List[float],     # |xₙ₊₁ - xₙ| at each step
    'function_values': List[float],   # f(x) at each step
    'error_message': Optional[str]    # Failure reason if applicable
}
```

## Error Handling and Edge Cases

### Numerical Stability Considerations

1. **Division by Zero Protection**:
   - Newton: Checks if |f'(x)| < ε before division
   - Secant: Checks if |f(xₙ) - f(xₙ₋₁)| < ε

2. **Convergence Failures**:
   - Maximum iteration limit prevents infinite loops
   - Methods return `converged=False` with error message

3. **Invalid Inputs**:
   - Bisection verifies sign change in initial interval
   - Type hints guide correct usage

4. **Floating-Point Precision**:
   - Default tolerances (1e-10) balance accuracy and achievability
   - Error history filtering removes inf/nan values before convergence rate estimation

### Common Failure Modes

| Method | Failure Mode | Cause | Mitigation |
|--------|-------------|-------|------------|
| Bisection | No sign change | f(a) × f(b) > 0 | Check interval before calling |
| Newton | Zero derivative | f'(x) ≈ 0 | Use larger epsilon, try different x₀ |
| Newton | Divergence | Poor initial guess | Choose x₀ closer to root |
| Secant | Division by zero | f(xₙ) ≈ f(xₙ₋₁) | Choose distinct initial guesses |
| All | Max iterations | Slow convergence | Increase max_iterations or adjust tolerances |

## Performance Characteristics

### Time Complexity

| Method | Per Iteration | Convergence Rate | Total Iterations (typical) |
|--------|---------------|------------------|----------------------------|
| Bisection | O(1) | Linear (p=1) | O(log₂(b-a)/δ) ≈ 30-50 |
| Newton | O(1) + derivative | Quadratic (p=2) | O(log log(1/δ)) ≈ 4-8 |
| Secant | O(1) | Superlinear (p≈1.618) | O(log log(1/δ)) ≈ 6-12 |

### Space Complexity

- **Memory Usage**: O(n) where n = number of iterations
  - Stores complete iteration history for visualization
  - Could be optimized to O(1) if only final result needed

- **Convergence Rate Estimation**: O(n) for storing error history

## Extensibility

### Adding New Test Functions

1. Add function and derivative to `test_functions.py`:
```python
def f4(x: float) -> float:
    return x**4 - 2*x**2 + 1

def df4(x: float) -> float:
    return 4*x**3 - 4*x
```

2. Register in `FUNCTIONS` dictionary:
```python
FUNCTIONS["f4"] = {
    "function": f4,
    "derivative": df4,
    "display_name": "f₄(x) = x⁴ - 2x² + 1",
    # ... other metadata
}
```

3. Function automatically appears in UI dropdown

### Adding New Numerical Methods

1. Implement in `numerical_methods.py` following the standard result format
2. Add method call in `run_root_finding_methods()` in `streamlit_app.py`
3. Update visualization and report modules as needed

## Technology Stack Details

### Core Dependencies

- **Python 3.11+**: Modern Python features (type hints, f-strings, dataclasses)
- **Streamlit ≥1.24.0**: Web framework with reactive UI updates
- **NumPy ≥1.24.0**: Numerical operations, vectorized calculations
- **Matplotlib ≥3.7.0**: Professional-quality plotting
- **Pandas ≥2.0.0**: Tabular data formatting

### Deployment Architecture

**Streamlit Community Cloud**:
- Automatic deployment from GitHub repository
- Serverless architecture, scales automatically
- HTTPS enabled by default
- Free hosting for public repositories

**DevContainer**:
- Consistent development environment across machines
- Pre-configured Python 3.11 with all dependencies
- VS Code integration with Pylance for type checking

## Security Considerations

1. **Input Validation**: User inputs are validated (numeric types, reasonable ranges)
2. **No External Data**: Application doesn't access external databases or APIs
3. **No User Data Storage**: All computation is session-based, no persistence
4. **Safe Evaluation**: No `eval()` or `exec()` of user input; only predefined functions

## Future Architecture Improvements

Potential enhancements that maintain current architecture:

1. **Caching**: Use `@st.cache_data` for expensive computations
2. **Parallel Execution**: Run three methods concurrently using threading
3. **Result Export**: Add CSV/JSON export of iteration data
4. **Plot Customization**: Allow user control of plot styling
5. **Method Plugins**: Plugin architecture for third-party methods
6. **Performance Profiling**: Add timing metrics for method comparison
