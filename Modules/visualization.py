"""
Visualization Module

This module provides functions for visualizing the numerical methods,
their convergence, and comparisons between different methods.

Functions include:
- Plotting test functions
- Visualizing iteration steps
- Plotting error convergence
- Comparing convergence rates across methods
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Callable, Dict, List, Any, Tuple, Optional
import pandas as pd


def plot_function(
    f: Callable[[float], float],
    x_range: Tuple[float, float],
    title: str = "Function Plot",
    x_label: str = "x",
    y_label: str = "f(x)",
    root: Optional[float] = None,
    points: Optional[List[Dict[str, float]]] = None,
    num_points: int = 1000,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot a function over a specified range, optionally marking the root and iteration points.
    
    Args:
        f: The function to plot
        x_range: Tuple (x_min, x_max) specifying the range of x values
        title: Plot title
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        root: Optional root value to mark on the plot
        points: Optional list of points from iterations to mark
        num_points: Number of points to calculate for the function plot
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate x values and compute corresponding y values
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = np.array([f(x) for x in x_vals])
    
    # Plot the function
    ax.plot(x_vals, y_vals, 'b-', label=f"f(x)")
    
    # Plot the x-axis
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Mark the root if provided
    if root is not None:
        root_y = f(root)
        ax.plot(root, root_y, 'ro', markersize=8, label=f"Root: x ≈ {root:.8f}")
        ax.plot([root, root], [0, root_y], 'r--', alpha=0.5)
    
    # Mark iteration points if provided
    if points is not None:
        x_points = [p.get('x', p.get('a', p.get('c', 0))) for p in points]
        y_points = [f(x) for x in x_points]
        
        # Use a colormap to indicate progression
        colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
        
        for i, (x, y, color) in enumerate(zip(x_points, y_points, colors)):
            if i == 0:
                ax.plot(x, y, 'o', color=color, markersize=6, alpha=0.7, label="Iterations")
            else:
                ax.plot(x, y, 'o', color=color, markersize=6, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def plot_error_convergence(
    error_history: List[float],
    title: str = "Error Convergence",
    method_name: str = "Method",
    figsize: Tuple[int, int] = (10, 6),
    use_log_scale: bool = True,
    rate: Optional[float] = None
):
    """
    Plot the error convergence over iterations.
    
    Args:
        error_history: List of error values for each iteration
        title: Plot title
        method_name: Name of the method being visualized
        figsize: Figure size as (width, height) in inches
        use_log_scale: Whether to use logarithmic scale for the y-axis
        rate: Optional convergence rate to display
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = range(len(error_history))
    
    # Plot the error convergence
    ax.plot(iterations, error_history, 'bo-', markersize=4, label=method_name)
    
    # Set logarithmic scale if requested
    if use_log_scale and min(error_history) > 0:
        ax.set_yscale('log')
    
    # Add convergence rate information if available
    if rate is not None:
        ax.text(0.05, 0.95, f"Convergence Rate: {rate:.4f}",
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Set labels and title
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error (log scale)" if use_log_scale else "Error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def compare_convergence_rates(
    results: Dict[str, Dict[str, Any]],
    title: str = "Convergence Rate Comparison",
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Compare the convergence rates of different methods.
    
    Args:
        results: Dictionary with method names as keys and result dictionaries as values
        title: Plot title
        figsize: Figure size as (width, height) in inches
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot error history for each method
    for method_name, result in results.items():
        if 'error_history' in result and result['error_history']:
            iterations = range(len(result['error_history']))
            ax.plot(iterations, result['error_history'], 'o-', markersize=4, label=method_name)
    
    # Set logarithmic scale for better visualization
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def create_iteration_table(iterations: List[Dict[str, Any]], method: str):
    """
    Create a formatted table of iteration details.
    
    Args:
        iterations: List of dictionaries with iteration details
        method: Name of the method (bisection, newton, secant)
    
    Returns:
        Pandas DataFrame with formatted iteration data
    """
    if not iterations:
        return pd.DataFrame()
    
    # Extract relevant columns based on the method
    if method == "bisection":
        df = pd.DataFrame([{
            'Iteration': it['iteration'],
            'Left (a)': it['a'],
            'Right (b)': it['b'],
            'Midpoint (c)': it.get('c'),
            'f(c)': it.get('w'),
            'Error': it.get('e')
        } for it in iterations])
    
    elif method == "newton":
        df = pd.DataFrame([{
            'Iteration': it['iteration'],
            'x': it['x'],
            'f(x)': it['fx'],
            'f\'(x)': it.get('fp'),
            'Step (d)': it.get('d')
        } for it in iterations])
    
    elif method == "secant":
        df = pd.DataFrame([{
            'Iteration': it['iteration'],
            'x': it.get('a'),
            'f(x)': it.get('fa'),
            'Step (d)': it.get('d')
        } for it in iterations])
    
    else:
        return pd.DataFrame()
    
    # Format floating-point numbers
    for col in df.columns:
        if col != 'Iteration':
            df[col] = df[col].apply(lambda x: f"{x:.10e}" if isinstance(x, (float, np.floating)) else x)
    
    return df


def plot_function_with_iterations_animation(
    f: Callable[[float], float],
    iterations: List[Dict[str, Any]],
    method: str,
    x_range: Tuple[float, float],
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create an animated visualization of the iteration process.
    
    For Streamlit, this creates a series of plots that can be cycled through
    using a slider to simulate animation.
    
    Args:
        f: The function to plot
        iterations: List of dictionaries with iteration details
        method: Method name (bisection, newton, secant)
        x_range: Tuple (x_min, x_max) specifying the range of x values
        figsize: Figure size as (width, height) in inches
    
    Returns:
        List of figures for each iteration stage
    """
    num_iterations = len(iterations)
    figures = []
    
    # Generate the base x values for the function plot
    x_vals = np.linspace(x_range[0], x_range[1], 1000)
    y_vals = np.array([f(x) for x in x_vals])
    
    for i in range(1, num_iterations):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the function
        ax.plot(x_vals, y_vals, 'b-', label=f"f(x)")
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Get current and previous iterations
        current = iterations[i]
        prev_iterations = iterations[:i]
        
        # Plot based on method
        if method == "bisection":
            # Plot current interval
            a, b = current['a'], current['b']
            c = current['c']
            fc = current.get('w', f(c))
            
            # Plot the interval
            ax.plot([a, a], [0, f(a)], 'g--', alpha=0.5)
            ax.plot([b, b], [0, f(b)], 'r--', alpha=0.5)
            
            # Plot the midpoint
            ax.plot(c, fc, 'mo', markersize=8, label=f"Iteration {i}: c = {c:.6f}")
            ax.plot([c, c], [0, fc], 'm--', alpha=0.5)
            
            # Mark the previous midpoints
            for j, prev in enumerate(prev_iterations[1:], 1):  # Skip the first entry which has no midpoint
                prev_c = prev.get('c')
                if prev_c is not None:
                    prev_fc = prev.get('w', f(prev_c))
                    ax.plot(prev_c, prev_fc, 'o', color='gray', alpha=0.5, markersize=5)
            
            ax.set_title(f"Bisection Method - Iteration {i}")
            
        elif method == "newton":
            # Plot current point
            x = current['x']
            fx = current['fx']
            fp = current['fp']
            
            # Calculate tangent line
            if fp is not None:
                # Tangent line: y - f(x) = f'(x)(t - x)
                # y = f(x) + f'(x)(t - x)
                tangent_x = np.linspace(x - 1, x + 1, 100)
                tangent_y = fx + fp * (tangent_x - x)
                ax.plot(tangent_x, tangent_y, 'g-', label="Tangent")
            
            # Plot current point
            ax.plot(x, fx, 'ro', markersize=8, label=f"Iteration {i}: x = {x:.6f}")
            
            # Plot line to x-axis (next guess)
            if fp is not None and abs(fp) > 1e-10:
                next_x = x - fx / fp
                ax.plot([x, next_x], [fx, 0], 'r--', alpha=0.5)
                ax.plot(next_x, 0, 'gx', markersize=8, label=f"Next x = {next_x:.6f}")
            
            # Mark previous points
            for j, prev in enumerate(prev_iterations):
                prev_x = prev.get('x')
                if prev_x is not None:
                    prev_fx = prev.get('fx', f(prev_x))
                    ax.plot(prev_x, prev_fx, 'o', color='gray', alpha=0.5, markersize=5)
            
            ax.set_title(f"Newton's Method - Iteration {i}")
            
        elif method == "secant":
            # For secant, we need the current and previous points
            if i >= 2:
                current_x = current.get('a')
                current_fx = current.get('fa', f(current_x))
                
                prev = iterations[i-1]
                prev_x = prev.get('a')
                prev_fx = prev.get('fa', f(prev_x))
                
                # Plot the secant line
                ax.plot([prev_x, current_x], [prev_fx, current_fx], 'g-', label="Secant")
                
                # Calculate next point (where secant line crosses x-axis)
                if prev_fx != current_fx:
                    slope = (current_fx - prev_fx) / (current_x - prev_x)
                    next_x = current_x - current_fx / slope
                    ax.plot([current_x, next_x], [current_fx, 0], 'r--', alpha=0.5)
                    ax.plot(next_x, 0, 'gx', markersize=8, label=f"Next x = {next_x:.6f}")
            
            # Plot current point
            current_x = current.get('a')
            current_fx = current.get('fa', f(current_x))
            ax.plot(current_x, current_fx, 'ro', markersize=8, label=f"Iteration {i}: x = {current_x:.6f}")
            
            # Mark all previous points
            for j, prev in enumerate(prev_iterations):
                prev_x = prev.get('a')
                if prev_x is not None:
                    prev_fx = prev.get('fa', f(prev_x))
                    ax.plot(prev_x, prev_fx, 'o', color='gray', alpha=0.5, markersize=5)
            
            ax.set_title(f"Secant Method - Iteration {i}")
        
        # Set labels and display options
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        figures.append(fig)
    
    return figures
