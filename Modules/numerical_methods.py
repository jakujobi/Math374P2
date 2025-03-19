"""
Numerical Methods for Root Finding

This module implements three numerical methods for finding roots of nonlinear equations:
1. Bisection Method - A reliable method that works by repeatedly bisecting an interval
2. Newton's Method - A fast-converging method that uses derivatives
3. Secant Method - A derivative-free method similar to Newton's but uses finite differences

Each method follows the pseudocode provided in the project description
and includes functionality for tracking iterations, errors, and convergence.

The module also provides a utility function for estimating convergence rates
based on the error history of a numerical method.

Implementation Notes:
- All methods return detailed dictionaries with iteration history and convergence information
- Special care is taken to handle potential numerical issues (division by near-zero values, etc.)
- Error history is tracked for convergence rate estimation

Project Information:
- Project 2 for Math 374: Scientific Computation (Spring 2025)
- South Dakota State University
- Developed by: John Akujobi (github.com/jakujobi)
- Website: jakujobi.com
- Professor: Dr. Jung-Han Kimn
"""

import numpy as np
from typing import Callable, Tuple, List, Dict, Any, Optional


def bisection_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    delta: float = 1e-10,
    epsilon: float = 1e-10,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    Implements the bisection method for finding roots of a nonlinear equation.
    
    The bisection method works by repeatedly dividing an interval in half and
    selecting the subinterval where the root must lie. It is based on the
    Intermediate Value Theorem: if a continuous function changes sign over an
    interval, it must have a root in that interval.
    
    Args:
        f: The function for which to find roots
        a: Left endpoint of the initial interval
        b: Right endpoint of the initial interval
        delta: Tolerance for the interval width (stopping criterion δ₁)
        epsilon: Tolerance for the function value (stopping criterion δ₂)
        max_iterations: Maximum number of iterations
        
    Returns:
        A dictionary containing:
            - 'root': The approximate root
            - 'iterations': List of dictionaries with iteration details
            - 'converged': Boolean indicating whether the method converged
            - 'iterations_count': Number of iterations performed
            - 'error_history': List of error values at each iteration
            - 'function_values': List of function values at each iteration
            - 'error_message': Description of the error (if convergence failed)
    
    Mathematical Details:
        - Order of convergence: Linear (order 1)
        - Guaranteed to converge if f is continuous and f(a) and f(b) have opposite signs
        - Each iteration reduces the interval size by half
    """
    # Compute initial function values
    u = f(a)
    v = f(b)
    e = b - a
    
    # Check if there's a sign change in the interval
    if np.sign(u) == np.sign(v):
        return {
            'root': None,
            'iterations': [],
            'converged': False,
            'iterations_count': 0,
            'error_history': [],
            'function_values': [],
            'error_message': "No sign change in the interval"
        }
    
    iterations = []
    error_history = [e]
    function_values = []
    
    # Record initial state
    iterations.append({
        'iteration': 0,
        'a': a,
        'b': b,
        'u': u,
        'v': v,
        'e': e,
        'c': None,
        'w': None
    })
    
    # Perform iterations
    for k in range(1, max_iterations + 1):
        # Update error and midpoint
        e = e / 2
        c = a + e
        w = f(c)
        
        function_values.append(w)
        
        # Record current iteration
        iterations.append({
            'iteration': k,
            'a': a,
            'b': b,
            'c': c,
            'u': u,
            'v': v,
            'w': w,
            'e': e
        })
        
        # Check convergence
        if abs(e) < delta or abs(w) < epsilon:
            return {
                'root': c,
                'iterations': iterations,
                'converged': True,
                'iterations_count': k,
                'error_history': error_history,
                'function_values': function_values
            }
        
        # Update interval
        if np.sign(w) != np.sign(u):
            b = c
            v = w
        else:
            a = c
            u = w
        
        error_history.append(abs(e))
    
    # If max iterations reached without convergence
    return {
        'root': c,
        'iterations': iterations,
        'converged': False,
        'iterations_count': max_iterations,
        'error_history': error_history,
        'function_values': function_values,
        'error_message': "Maximum iterations reached"
    }


def newton_method(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    delta1: float = 1e-10,
    delta2: float = 1e-10,
    epsilon: float = 1e-10,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    Implements Newton's method for finding roots of a nonlinear equation.
    
    Args:
        f: The function for which to find roots
        df: The derivative of the function f
        x0: Initial guess for the root
        delta1: Tolerance for the step size
        delta2: Tolerance for the function value
        epsilon: Tolerance for the derivative value (to avoid division by zero)
        max_iterations: Maximum number of iterations
        
    Returns:
        A dictionary containing:
            - 'root': The approximate root
            - 'iterations': List of dictionaries with iteration details
            - 'converged': Boolean indicating whether the method converged
            - 'iterations_count': Number of iterations performed
            - 'error_history': List of error values at each iteration
            - 'function_values': List of function values at each iteration
    """
    x = x0
    fx = f(x)
    
    iterations = []
    error_history = []
    function_values = [fx]
    
    # Record initial state
    iterations.append({
        'iteration': 0,
        'x': x,
        'fx': fx,
        'fp': None,
        'd': None
    })
    
    # Perform iterations
    for k in range(1, max_iterations + 1):
        # Compute derivative
        fp = df(x)
        
        # Check if derivative is too small
        if abs(fp) < epsilon:
            return {
                'root': x,
                'iterations': iterations,
                'converged': False,
                'iterations_count': k,
                'error_history': error_history,
                'function_values': function_values,
                'error_message': "Small derivative encountered"
            }
        
        # Compute update step
        d = fx / fp
        x_new = x - d
        x = x_new
        fx = f(x)
        
        function_values.append(fx)
        error_history.append(abs(d))
        
        # Record current iteration
        iterations.append({
            'iteration': k,
            'x': x,
            'fx': fx,
            'fp': fp,
            'd': d
        })
        
        # Check convergence
        if abs(d) < delta1 or abs(fx) < delta2:
            return {
                'root': x,
                'iterations': iterations,
                'converged': True,
                'iterations_count': k,
                'error_history': error_history,
                'function_values': function_values
            }
    
    # If max iterations reached without convergence
    return {
        'root': x,
        'iterations': iterations,
        'converged': False,
        'iterations_count': max_iterations,
        'error_history': error_history,
        'function_values': function_values,
        'error_message': "Maximum iterations reached"
    }


def secant_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    delta1: float = 1e-10,
    delta2: float = 1e-10,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    Implements the secant method for finding roots of a nonlinear equation.
    
    Args:
        f: The function for which to find roots
        a: First initial guess
        b: Second initial guess
        delta1: Tolerance for the step size
        delta2: Tolerance for the function value
        max_iterations: Maximum number of iterations
        
    Returns:
        A dictionary containing:
            - 'root': The approximate root
            - 'iterations': List of dictionaries with iteration details
            - 'converged': Boolean indicating whether the method converged
            - 'iterations_count': Number of iterations performed
            - 'error_history': List of error values at each iteration
            - 'function_values': List of function values at each iteration
    """
    # Initialize
    fa = f(a)
    fb = f(b)
    
    iterations = []
    error_history = []
    function_values = [fa, fb]
    
    # Swap if |fb| < |fa|
    if abs(fb) < abs(fa):
        a, b = b, a
        fa, fb = fb, fa
    
    # Record initial states
    iterations.append({
        'iteration': 0,
        'a': a,
        'fa': fa,
        'd': None
    })
    
    iterations.append({
        'iteration': 1,
        'a': b,
        'fa': fb,
        'd': None
    })
    
    # Main iteration loop
    for k in range(2, max_iterations + 2):
        # Swap if |fb| < |fa|
        if abs(fb) < abs(fa):
            a, b = b, a
            fa, fb = fb, fa
        
        # Compute the update
        if abs(fb - fa) < 1e-15:  # More robust check for near-zero division
            return {
                'root': a,
                'iterations': iterations,
                'converged': False,
                'iterations_count': k-1,
                'error_history': error_history,
                'function_values': function_values,
                'error_message': "Division by zero encountered (function values too close)"
            }
            
        d = (b - a) / (fb - fa) * fa
        b = a
        fb = fa
        a = a - d
        fa = f(a)
        
        function_values.append(fa)
        error_history.append(abs(d))
        
        # Record current iteration
        iterations.append({
            'iteration': k,
            'a': a,
            'fa': fa,
            'd': d
        })
        
        # Check convergence
        if abs(d) < delta1 or abs(fa) < delta2:
            return {
                'root': a,
                'iterations': iterations,
                'converged': True,
                'iterations_count': k,
                'error_history': error_history,
                'function_values': function_values
            }
    
    # If max iterations reached without convergence
    return {
        'root': a,
        'iterations': iterations,
        'converged': False,
        'iterations_count': max_iterations,
        'error_history': error_history,
        'function_values': function_values,
        'error_message': "Maximum iterations reached"
    }


def estimate_convergence_rate(error_history: List[float]) -> Optional[float]:
    """
    Estimates the convergence rate of a numerical method based on the error history.
    
    For a method with convergence rate p, we expect:
    |e_{k+1}| ≈ C * |e_k|^p for some constant C
    
    Taking logarithms:
    log|e_{k+1}| ≈ log(C) + p * log|e_k|
    
    So the slope of log|e_{k+1}| vs log|e_k| gives us p.
    
    Args:
        error_history: List of error values from each iteration
        
    Returns:
        Estimated convergence rate or None if estimation is not possible
    """
    if len(error_history) < 3:
        return None
    
    # Skip the first few iterations to avoid initial irregularities
    start_idx = max(1, len(error_history) // 5)  # Skip initial 20% of iterations
    
    log_errors = np.log10(np.array(error_history[start_idx:]))
    log_errors_prev = np.log10(np.array(error_history[start_idx-1:-1]))
    
    # Filter out invalid values (like inf or nan)
    valid_indices = np.logical_and(
        np.isfinite(log_errors),
        np.isfinite(log_errors_prev)
    )
    
    if np.sum(valid_indices) < 3:
        return None
    
    log_errors = log_errors[valid_indices]
    log_errors_prev = log_errors_prev[valid_indices]
    
    # Linear regression to find the slope
    try:
        # Use polyfit to get the slope of the linear regression
        slope, _ = np.polyfit(log_errors_prev, log_errors, 1)
        return slope
    except (np.linalg.LinAlgError, ValueError) as e:
        # Catch specific errors related to linear algebra problems or invalid values
        return None
