"""
Test Functions Module

This module defines the test functions and their derivatives for the root-finding methods.
These functions are used to evaluate the performance of numerical methods.

The module includes three primary test functions with diverse properties:

1. f1(x) = x^2 - 4*sin(x)
   - Combines polynomial and trigonometric components
   - Has multiple roots and interesting behavior
   - Useful for testing methods on more complex functions

2. f2(x) = x^2 - 1
   - Simple quadratic function with analytical roots at x = ±1
   - Useful as a baseline test case with well-understood behavior
   - Provides predictable convergence patterns

3. f3(x) = x^3 - 3*x^2 + 3*x - 1
   - Cubic polynomial with a triple root at x = 1
   - Challenging for numerical methods due to the multiplicity of the root
   - Tests the robustness of methods when faced with nearly-flat regions

Each function has a corresponding derivative function for use with Newton's method.
The module also provides utility functions to retrieve information about these functions.
"""

import numpy as np
from typing import Callable, Dict, Tuple, List


def f1(x: float) -> float:
    """
    Function f1(x) = x^2 - 4*sin(x)
    
    This function combines a polynomial term (x^2) with a trigonometric term (-4*sin(x)).
    It has multiple roots and exhibits interesting behavior that makes it suitable
    for testing numerical methods.
    
    Args:
        x: Input value
        
    Returns:
        Function value at x
    """
    return x**2 - 4*np.sin(x)


def df1(x: float) -> float:
    """
    Derivative of f1(x) = x^2 - 4*sin(x)
    f1'(x) = 2*x - 4*cos(x)
    
    This is the analytical derivative of f1, used in Newton's method.
    
    Args:
        x: Input value
        
    Returns:
        Derivative value at x
    """
    return 2*x - 4*np.cos(x)


def f2(x: float) -> float:
    """
    Function f2(x) = x^2 - 1
    
    Args:
        x: Input value
        
    Returns:
        Function value at x
    """
    return x**2 - 1


def df2(x: float) -> float:
    """
    Derivative of f2(x) = x^2 - 1
    f2'(x) = 2*x
    
    Args:
        x: Input value
        
    Returns:
        Derivative value at x
    """
    return 2*x


def f3(x: float) -> float:
    """
    Function f3(x) = x^3 - 3*x^2 + 3*x - 1
    
    Args:
        x: Input value
        
    Returns:
        Function value at x
    """
    return x**3 - 3*x**2 + 3*x - 1


def df3(x: float) -> float:
    """
    Derivative of f3(x) = x^3 - 3*x^2 + 3*x - 1
    f3'(x) = 3*x^2 - 6*x + 3
    
    Args:
        x: Input value
        
    Returns:
        Derivative value at x
    """
    return 3*x**2 - 6*x + 3


# Dictionary mapping function IDs to function objects and their derivatives
FUNCTIONS = {
    "f1": {
        "function": f1,
        "derivative": df1,
        "display_name": "f₁(x) = x² - 4sin(x)",
        "latex": r"f_1(x) = x^2 - 4\sin(x)",
        "description": "Combination of polynomial and trigonometric terms",
        "known_roots": [0],  # Known roots for validation
        "suggested_intervals": [(-3, 3), (1, 3)]
    },
    "f2": {
        "function": f2,
        "derivative": df2,
        "display_name": "f₂(x) = x² - 1",
        "latex": r"f_2(x) = x^2 - 1",
        "description": "Simple quadratic function with analytical roots at x = ±1",
        "known_roots": [-1, 1],
        "suggested_intervals": [(-2, 0), (0, 2)]
    },
    "f3": {
        "function": f3,
        "derivative": df3,
        "display_name": "f₃(x) = x³ - 3x² + 3x - 1",
        "latex": r"f_3(x) = x^3 - 3x^2 + 3x - 1",
        "description": "Cubic polynomial with a single root at x = 1",
        "known_roots": [1],
        "suggested_intervals": [(0, 2)]
    }
}


def get_function_details(function_id: str) -> Dict:
    """
    Get the details for a specific function by ID.
    
    Args:
        function_id: The ID of the function to retrieve
        
    Returns:
        A dictionary with function details
        
    Raises:
        ValueError: If the function_id is not found
    """
    if function_id not in FUNCTIONS:
        raise ValueError(f"Function ID '{function_id}' not found")
    
    return FUNCTIONS[function_id]


def get_all_functions() -> Dict:
    """
    Get all available test functions.
    
    Returns:
        Dictionary of all function details
    """
    return FUNCTIONS
