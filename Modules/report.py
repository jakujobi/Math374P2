"""
Report Module

This module provides the content and structure for the detailed mathematical report
on root-finding methods, including explanations of algorithms, termination criteria,
convergence rates, and comparative analysis.
"""

import streamlit as st
from typing import Dict, Any, List, Optional


def render_introduction():
   """Render the introduction section of the report."""
   st.markdown("""
   # Root-Finding Methods: Analysis and Comparison
   
   This report presents an analysis of three numerical methods for finding roots of nonlinear equations:
   the **Bisection Method**, **Newton's Method**, and the **Secant Method**. We examine their
   implementation, termination criteria, convergence rates, and comparative performance
   on three different test functions.
   
   ---
   """)


def render_methods_explanation():
   """Render the explanation of numerical methods."""
   st.markdown("""
   ## Numerical Methods Overview
   
   ### Bisection Method
   
   The bisection method is a simple and robust root-finding algorithm based on the 
   Intermediate Value Theorem. If a continuous function changes sign over an interval,
   then it must have at least one root within that interval.
   
   #### Algorithm:
   1. Start with an interval [a, b] where f(a) and f(b) have opposite signs
   2. Compute the midpoint c = (a + b) / 2 and evaluate f(c)
   3. If f(c) is sufficiently close to zero or the interval is sufficiently small, return c as the root
   4. Otherwise, replace either [a, c] or [c, b] as the new interval, depending on where the sign change occurs
   5. Repeat steps 2-4 until convergence
   
   #### Pseudocode:
   ```
   1. Compute:
      u ← f(a)
      v ← f(b)
      e ← b - a
   
   2. If sign(u) = sign(v), then:
         Stop (no guaranteed root in interval)
   
   3. For k = 1 to M do:
      a. Update error:
            e ← e / 2
            c ← a + e
            w ← f(c)
   
      b. Check convergence:
            If |e| < δ₁ or |w| < δ₂ then:
               Stop (convergence criteria met)
   
      c. Check sign change:
            If sign(w) ≠ sign(u), then:
               b ← c
               v ← w
            Else:
               a ← c
               u ← w
   ```
   
   ### Newton's Method
   
   Newton's method (also known as the Newton-Raphson method) uses function derivatives
   to quickly converge to a root. It iteratively improves an approximation by finding the
   intersection of the tangent line with the x-axis.
   
   #### Algorithm:
   1. Start with an initial guess x₀
   2. Compute the function value f(xₙ) and its derivative f'(xₙ)
   3. Update the approximation: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
   4. Repeat steps 2-3 until convergence
   
   #### Pseudocode:
   ```
   1. Compute:
      fx ← f(x)
   
   2. For k = 1 to nmax do:
      a. Compute derivative:
            fp ← f'(x)
   
      b. Check if derivative is too small:
            If |fp| < ε then:
               Return (small derivative - potential issue)
   
      c. Compute update step:
            d ← fx / fp
            x ← x - d
            fx ← f(x)
   
      d. Check convergence:
            If |d| < δ₁ or |fx| < δ₂ then:
               Return x (convergence criteria met)
   ```
   
   ### Secant Method
   
   The secant method is similar to Newton's method but eliminates the need for 
   derivatives by approximating the derivative with a finite difference using the
   two most recent approximations.
   
   #### Algorithm:
   1. Start with two initial guesses x₀ and x₁
   2. Compute the secant approximation: xₙ₊₁ = xₙ - f(xₙ) * (xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁))
   3. Repeat until convergence
   
   #### Pseudocode:
   ```
   1. Initialize:
      fa ← f(a)
      fb ← f(b)
   
   2. If |fb| < |fa|, then:
         Swap a and b
         Swap fa and fb
   
   3. For k = 2 to nmax do:
      a. If |fb| < |fa|, then:
            Swap a and b
            Swap fa and fb
   
      b. Compute:
            d ← (b - a) / (fb - fa)
            b ← a
            fb ← fa
            d ← d * fa
            a ← a - d
            fa ← f(a)
   
      c. Check convergence:
            If |d| < δ₁ or |fa| < δ₂ then:
               Return a (convergence criteria met)
   ```
   """)


def render_termination_criteria():
   """Render the explanation of termination criteria."""
   st.markdown("""
   ## Termination Criteria
   
   Proper termination criteria are essential for numerical methods to efficiently converge to an accurate solution.
   For our implementation, we use a combination of criteria:
   
   ### 1. Step Size Tolerance (δ₁)
   
   This criterion checks if the change in the approximation is sufficiently small:
   
   - In the **Bisection Method**: |b - a| < δ₁ (the interval width is small enough)
   - In **Newton's Method**: |xₙ₊₁ - xₙ| < δ₁ (the change in successive approximations is small)
   - In the **Secant Method**: |xₙ₊₁ - xₙ| < δ₁ (similar to Newton's method)
   
   ### 2. Function Value Tolerance (δ₂)
   
   This criterion checks if the function value at the current approximation is close enough to zero:
   
   - |f(x)| < δ₂
   
   ### 3. Maximum Iterations
   
   To prevent infinite loops in cases where convergence is slow or fails, a maximum number of iterations is set.
   
   ### 4. Additional Safeguards
   
   - For **Newton's Method**: An additional check |f'(x)| < ε ensures we avoid division by near-zero derivatives.
   - For the **Secant Method**: Swapping points based on function value magnitudes improves numerical stability.
   
   ### Why Use Multiple Criteria?
   
   Using multiple termination criteria provides a robust stopping mechanism that balances:
   
   1. **Accuracy**: Ensures the approximation is sufficiently close to a true root
   2. **Efficiency**: Prevents unnecessary iterations when an acceptable solution is found
   3. **Reliability**: Safeguards against pathological cases where one criterion might fail
   
   In practice, we set δ₁ and δ₂ to small values (typically 10⁻¹⁰) to achieve high precision while
   maintaining reasonable computational efficiency.
   """)


def render_convergence_rates():
   """Render the explanation of convergence rates."""
   st.markdown("""
   ## Convergence Rates
   
   The **convergence rate** of a numerical method describes how quickly the error decreases as the iterations progress.
   For a sequence of approximations {xₙ} converging to a root α, if there exists a constant C > 0 and p > 0 such that:
   
   |xₙ₊₁ - α| ≤ C|xₙ - α|ᵖ
   
   then p is the **order of convergence**.
   
   ### Theoretical Convergence Rates
   
   1. **Bisection Method**: Linear convergence (p = 1) with a constant C = 1/2
   
      The error is halved with each iteration: |eₙ₊₁| ≤ (1/2)|eₙ|
   
   2. **Newton's Method**: Quadratic convergence (p = 2) near the root
   
      When close to the root, the error is squared: |eₙ₊₁| ≤ C|eₙ|²
   
   3. **Secant Method**: Superlinear convergence (p ≈ 1.618, the golden ratio)
   
      The convergence rate is between linear and quadratic: |eₙ₊₁| ≤ C|eₙ|^(1.618)
   
   ### Estimating Convergence Rates from Data
   
   We can estimate the convergence rate from the generated sequence of errors.
   For consecutive errors eₙ, eₙ₊₁, eₙ₊₂, we have:
   
   |eₙ₊₁| ≈ C|eₙ|ᵖ
   |eₙ₊₂| ≈ C|eₙ₊₁|ᵖ
   
   Taking logarithms and solving for p:
   
   p ≈ log(|eₙ₊₂|/|eₙ₊₁|) / log(|eₙ₊₁|/|eₙ|)
   
   In practice, we use linear regression on the log-log plot of consecutive errors to estimate p.
   """)


def render_method_comparison(results: Dict[str, Dict[str, Any]] = None):
   """
   Render a comparison of the different methods.
   
   Args:
      results: Optional dictionary with method results for displaying actual comparison data
   """
   st.markdown("""
   ## Method Comparison
   
   Each numerical method has distinct characteristics that make it more or less suitable for different scenarios:
   
   ### Bisection Method
   
   **Advantages**:
   - Guaranteed to converge if the initial interval contains a root
   - Simple implementation
   - Robust against pathological function behavior
   
   **Disadvantages**:
   - Slow convergence (linear)
   - Requires an initial interval with a sign change
   - May not find all roots if multiple exist in the interval
   
   **Best suited for**: Initial bracketing of roots or cases where reliability is more important than speed
   
   ### Newton's Method
   
   **Advantages**:
   - Very fast convergence near the root (quadratic)
   - Precise results with fewer iterations than bisection
   - Can converge to complex roots with complex initial values
   
   **Disadvantages**:
   - Requires the derivative of the function
   - Can diverge or oscillate with poor initial guesses
   - May fail near inflection points where f'(x) ≈ 0
   
   **Best suited for**: Problems where the derivative is readily available and a good initial guess can be made
   
   ### Secant Method
   
   **Advantages**:
   - Faster convergence than bisection (superlinear)
   - Does not require derivatives
   - Often nearly as fast as Newton's method in practice
   
   **Disadvantages**:
   - Can diverge with poor initial guesses
   - Less robust than bisection
   - Requires two initial points
   
   **Best suited for**: Problems where derivatives are difficult to compute or expensive
   """)
   
   if results:
      st.markdown("### Quantitative Comparison")
      
      # Create a comparison table
      data = []
      for method_name, result in results.items():
         data.append({
               "Method": method_name,
               "Root Found": f"{result.get('root', 'N/A'):.10e}" if result.get('root') is not None else "N/A",
               "Iterations": result.get('iterations_count', 'N/A'),
               "Converged": "Yes" if result.get('converged', False) else "No",
               "Final Error": f"{result.get('error_history', [0])[-1]:.10e}" if result.get('error_history') else "N/A",
               "Estimated Convergence Rate": f"{result.get('convergence_rate', 'N/A'):.4f}" if result.get('convergence_rate') is not None else "N/A"
         })
      
      st.table(data)


def render_function_analysis(function_id: str):
   """
   Render analysis of a specific test function.
   
   Args:
      function_id: ID of the function to analyze
   """
   function_details = {
      "f1": {
         "title": r"$f_1(x) = x^2 - 4\sin(x)$",
         "description": """
         This function combines polynomial and trigonometric terms. It has multiple roots
         due to the oscillatory nature of the sine function. The known root at x = 0 occurs
         because f₁(0) = 0² - 4sin(0) = 0, and there are additional roots where x² = 4sin(x).
         
         **Characteristics**:
         - Contains both polynomial and trigonometric components
         - Multiple roots (at x = 0 and where x² = 4sin(x))
         - Derivative: f₁'(x) = 2x - 4cos(x)
         """,
         "challenges": """
         **Challenges for numerical methods**:
         - The presence of multiple roots may cause methods to converge to different roots depending on initial conditions
         - Near certain values, the derivative approach zero, which can cause issues for Newton's method
         - The oscillatory nature requires careful selection of initial brackets for the bisection method
         """
      },
      "f2": {
         "title": r"$f_2(x) = x^2 - 1$",
         "description": """
         This is a simple quadratic function with well-known analytical roots at x = ±1.
         The function is symmetric about the y-axis.
         
         **Characteristics**:
         - Simple quadratic function
         - Two roots: x = -1 and x = 1
         - Derivative: f₂'(x) = 2x (linear)
         """,
         "challenges": """
         **Challenges for numerical methods**:
         - For Newton's method, starting at x = 0 leads to issues since f₂'(0) = 0
         - Otherwise, this function serves as a good benchmark due to its simplicity
         """
      },
      "f3": {
         "title": r"$f_3(x) = x^3 - 3x^2 + 3x - 1$",
         "description": """
         This is a cubic polynomial with a single root at x = 1 with multiplicity 3.
         This means (x-1)³ = x³ - 3x² + 3x - 1.
         
         **Characteristics**:
         - Cubic polynomial
         - Single root at x = 1 with multiplicity 3
         - Derivative: f₃'(x) = 3x² - 6x + 3 = 3(x-1)²
         """,
         "challenges": """
         **Challenges for numerical methods**:
         - The multiplicity of the root causes slower convergence for all methods
         - For Newton's method, the convergence rate may reduce from quadratic to linear due to the root's multiplicity
         - The function approaches zero very gradually near the root, requiring more precise tolerance settings
         """
      }
   }
   
   details = function_details.get(function_id, {
      "title": "Function Analysis",
      "description": "No detailed analysis available for this function.",
      "challenges": ""
   })
   
   st.markdown(f"## Analysis of {details['title']}")
   st.markdown(details["description"])
   st.markdown(details["challenges"])


def render_conclusion(results: Optional[Dict[str, Dict[str, Any]]] = None):
   """
   Render the conclusion section of the report.
   
   Args:
      results: Optional dictionary with method results for more specific conclusions
   """
   st.markdown("""
   ## Conclusion
   
   Our implementation and analysis of the bisection, Newton, and secant methods reveal important insights
   about numerical root-finding approaches:
   
   1. **Termination Criteria**: Using a combination of step size and function value tolerances provides
      a robust convergence check that balances accuracy and efficiency.
   
   2. **Convergence Rates**: The empirical convergence rates generally align with theoretical expectations:
      - Bisection Method: Linear convergence (p ≈ 1)
      - Newton's Method: Quadratic convergence (p ≈ 2) for simple roots
      - Secant Method: Superlinear convergence (p ≈ 1.6)
   
   3. **Method Selection**: The choice of method should consider:
      - Availability of derivatives
      - Quality of initial guesses
      - Need for reliability vs. speed
      - Function characteristics (e.g., multiple roots, regions with small derivatives)
   
   For practical applications, a hybrid approach is often beneficial—using the bisection method to
   obtain a reliable bracket, followed by Newton's or the secant method for faster final convergence.
   """)
   
   if results:
      # Add specific observations based on actual results
      best_method = min(results.items(), key=lambda x: x[1].get('iterations_count', float('inf')) if x[1].get('converged', False) else float('inf'))
      
      if best_method[1].get('converged', False):
         st.markdown(f"""
         ### Key Observations from Our Results
         
         - The **{best_method[0]}** method demonstrated the best performance on our test functions,
            converging in just {best_method[1].get('iterations_count')} iterations.
         
         - The observed convergence rates aligned with theoretical expectations, with some variations
            due to function characteristics and initial conditions.
         
         - Function characteristics significantly influence method performance:
            - Functions with multiple roots benefit from careful initial bracketing
            - Functions with regions where f'(x) ≈ 0 pose challenges for derivative-based methods
            - Roots with higher multiplicity typically result in slower convergence
         """)


def render_report(active_function: str = "f1", results: Dict[str, Dict[str, Any]] = None):
   """
   Render the complete mathematical report.
   
   Args:
      active_function: ID of the currently active function
      results: Dictionary with method results for the active function
   """
   render_introduction()
   render_methods_explanation()
   render_termination_criteria()
   render_convergence_rates()
   render_method_comparison(results)
   render_function_analysis(active_function)
   render_conclusion(results)
   
   # References section
   st.markdown("""
   ## References
   
   1. Burden, R. L., & Faires, J. D. (2010). *Numerical Analysis* (9th ed.). Brooks/Cole.
   
   2. Heath, M. T. (2018). *Scientific Computing: An Introductory Survey* (3rd ed.). SIAM.
   
   3. Atkinson, K. E. (1989). *An Introduction to Numerical Analysis* (2nd ed.). John Wiley & Sons.
   
   4. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).
      *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.
   """)


def render_pseudocode():
   """Render the pseudocode for all three methods."""
   st.markdown("""
   # Pseudocode for Numerical Methods
   
   ## Bisection Method
   
   ```
   Algorithm: Bisection Method
   
   INPUT:
   - Interval endpoints: a, b
   - Maximum iterations: M
   - Error tolerances: δ₁, δ₂
   
   VARIABLES:
   - u, v, w, e, c
   
   STEPS:
   1. Compute:
      u ← f(a)
      v ← f(b)
      e ← b - a
   
   2. If sign(u) = sign(v), then:
         Stop (no guaranteed root in interval)
   
   3. For k = 1 to M do:
      a. Update error:
            e ← e / 2
            c ← a + e
            w ← f(c)
   
      b. Check convergence:
            If |e| < δ₁ or |w| < δ₂ then:
               Stop (convergence criteria met)
   
      c. Check sign change:
            If sign(w) ≠ sign(u), then:
               b ← c
               v ← w
            Else:
               a ← c
               u ← w
   ```
   
   ## Newton's Method
   
   ```
   Algorithm: Newton's Method
   
   INPUT:
   - Function and its derivative: f, f'
   - Initial guess: x
   - Maximum iterations: nmax
   - Error tolerances: δ₁, δ₂, ε
   
   VARIABLES:
   - fx, fp, d
   
   STEPS:
   1. Compute:
      fx ← f(x)
   
   2. For k = 1 to nmax do:
      a. Compute derivative:
            fp ← f'(x)
   
      b. Check if derivative is too small:
            If |fp| < ε then:
               Return (small derivative - potential issue)
   
      c. Compute update step:
            d ← fx / fp
            x ← x - d
            fx ← f(x)
   
      d. Check convergence:
            If |d| < δ₁ or |fx| < δ₂ then:
               Return x (convergence criteria met)
   ```
   
   ## Secant Method
   
   ```
   Algorithm: Secant Method
   
   INPUT:
   - Function: f
   - Initial values: a, b
   - Maximum iterations: nmax
   - Convergence tolerances: δ₁, δ₂
   
   VARIABLES:
   - fa, fb, d
   
   STEPS:
   1. Initialize:
      fa ← f(a)
      fb ← f(b)
   
   2. If |fb| < |fa|, then:
         Swap a and b
         Swap fa and fb
   
   3. For k = 2 to nmax do:
      a. If |fb| < |fa|, then:
            Swap a and b
            Swap fa and fb
   
      b. Compute:
            d ← (b - a) / (fb - fa)
            b ← a
            fb ← fa
            d ← d * fa
            a ← a - d
            fa ← f(a)
   
      c. Check convergence:
            If |d| < δ₁ or |fa| < δ₂ then:
               Return a (convergence criteria met)
   ```
   """)
