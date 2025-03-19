"""
Root-Finding Methods Analysis

This is the main Streamlit application that integrates all modules to provide an
interactive interface for analyzing and comparing different numerical methods for
finding roots of nonlinear equations.

The application implements three methods:
1. Bisection Method
2. Newton's Method
3. Secant Method

It allows users to select test functions, configure method parameters, visualize results,
and analyze convergence rates.

Project Information:
- Project 2 for Math 374: Scientific Computation (Spring 2025)
- South Dakota State University
- Developed by: John Akujobi (github.com/jakujobi)
- Website: jakujobi.com
- Professor: Dr. Jung-Han Kimn
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Modules.numerical_methods import bisection_method, newton_method, secant_method, estimate_convergence_rate
from Modules.test_functions import get_function_details, get_all_functions
from Modules.visualization import (
    plot_function, plot_error_convergence, compare_convergence_rates,
    create_iteration_table, plot_function_with_iterations_animation
)
from Modules.report import render_report, render_pseudocode

# Set page configuration
st.set_page_config(
    page_title="Root-Finding Methods Analysis: Math 374 Project 2 - John Akujobi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4263eb;
    }
    .table-container {
        border-radius: 5px;
        overflow: auto;
        max-height: 400px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(240, 242, 246, 0.9);
        color: #262730;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        border-top: 1px solid #e6e9ef;
        z-index: 999;
    }
    .footer a {
        color: #0068c9;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Add footer with developer information
st.markdown(
    """
    <div class="footer">
        Project 2 for Math 374: Scientific Computation (Spring 2025) | 
        South Dakota State University | 
        Developed by <a href="https://github.com/jakujobi" target="_blank">John Akujobi</a> | 
        <a href="https://jakujobi.com" target="_blank">jakujobi.com</a> | 
        Professor: Dr. Jung-Han Kimn
    </div>
    """,
    unsafe_allow_html=True,
)

def run_root_finding_methods(function_id, settings):
    """
    Run all three root-finding methods on the selected function with specified settings.
    
    This is a core function that executes the numerical methods with the user-configured
    parameters and collects their results for analysis and visualization. Each method
    is run only if enabled in the settings, allowing for selective comparison.
    
    Process flow:
    1. Retrieve the function and its derivative from the function database
    2. Run each enabled numerical method with the specified settings
    3. Estimate convergence rates for each method based on error history
    4. Compile all results into a single dictionary keyed by method name
    
    Args:
        function_id: ID of the function to analyze (corresponds to an entry in test_functions.py)
        settings: Dictionary of settings for each method including:
            - bisection: Parameters for the bisection method
            - newton: Parameters for Newton's method
            - secant: Parameters for the secant method
            Each method's settings contain:
            - use: Boolean flag indicating whether to run this method
            - method-specific parameters like a, b, x0, tolerances, etc.
        
    Returns:
        Dictionary with results from each method, where keys are method names
        and values are dictionaries containing the method's output including:
        - root: Approximated root value
        - iterations: Detailed iteration history
        - converged: Whether the method converged
        - error_history: List of error values at each iteration
        - convergence_rate: Estimated order of convergence
    """
    # Get function details
    func_details = get_function_details(function_id)
    f = func_details["function"]
    df = func_details["derivative"]
    
    results = {}
    
    # Run Bisection Method
    if settings["bisection"]["use"]:
        a = settings["bisection"]["a"]
        b = settings["bisection"]["b"]
        delta = 10**settings["bisection"]["delta_exp"]
        epsilon = 10**settings["bisection"]["epsilon_exp"]
        max_iter = settings["bisection"]["max_iterations"]
        
        bisection_result = bisection_method(f, a, b, delta, epsilon, max_iter)
        bisection_result["convergence_rate"] = estimate_convergence_rate(bisection_result["error_history"])
        results["Bisection Method"] = bisection_result
    
    # Run Newton's Method
    if settings["newton"]["use"]:
        x0 = settings["newton"]["x0"]
        delta1 = 10**settings["newton"]["delta1_exp"]
        delta2 = 10**settings["newton"]["delta2_exp"]
        epsilon = 10**settings["newton"]["epsilon_exp"]
        max_iter = settings["newton"]["max_iterations"]
        
        newton_result = newton_method(f, df, x0, delta1, delta2, epsilon, max_iter)
        newton_result["convergence_rate"] = estimate_convergence_rate(newton_result["error_history"])
        results["Newton's Method"] = newton_result
    
    # Run Secant Method
    if settings["secant"]["use"]:
        a = settings["secant"]["a"]
        b = settings["secant"]["b"]
        delta1 = 10**settings["secant"]["delta1_exp"]
        delta2 = 10**settings["secant"]["delta2_exp"]
        max_iter = settings["secant"]["max_iterations"]
        
        secant_result = secant_method(f, a, b, delta1, delta2, max_iter)
        secant_result["convergence_rate"] = estimate_convergence_rate(secant_result["error_history"])
        results["Secant Method"] = secant_result
    
    return results


def main():
    """
    Main application function that sets up the Streamlit interface and handles user interaction.
    
    This function is responsible for:
    1. Setting up the sidebar for user input and configuration
        - Function selection with description and mathematical representation
        - Method-specific parameter controls with appropriate defaults
        - Advanced settings for fine-tuning method parameters
    
    2. Creating the main content area with multiple tabs
        - Analysis Dashboard: Interactive visualizations and result summaries
        - Detailed Report: Mathematical explanation of methods and analysis
        - Pseudocode: Algorithmic representation of each method
        - Code Report: Implementation details and architecture overview
    
    3. Managing application state and workflow
        - Storing results in session state for persistence between interactions
        - Retrieving and displaying previous results when available
        - Orchestrating the analysis workflow from input to visualization
    
    The function follows a reactive design pattern where:
    - User inputs in the sidebar trigger computations when "Run Analysis" is clicked
    - Results are stored in session state and displayed across multiple views
    - Each tab provides a different perspective on the same underlying data
    """
    # Sidebar with function selection and settings
    with st.sidebar:
        st.title("Settings")
        
        # Function selection
        st.header("Test Function")
        all_functions = get_all_functions()
        function_options = {func_id: details["display_name"] for func_id, details in all_functions.items()}
        selected_function = st.selectbox(
            "Select a function to analyze",
            options=list(function_options.keys()),
            format_func=lambda x: function_options[x]
        )
        
        # Get function details
        function_details = get_function_details(selected_function)
        
        # Display function info
        st.markdown(f"**Function**: {function_details['latex']}")
        st.markdown(f"**Description**: {function_details['description']}")
        
        # Settings for numerical methods
        st.header("Numerical Methods")
        
        # Bisection Method settings
        st.subheader("Bisection Method")
        use_bisection = st.checkbox("Use Bisection Method", value=True)
        
        bisection_settings = {}
        if use_bisection:
            suggested_intervals = function_details["suggested_intervals"]
            selected_interval = st.selectbox(
                "Select interval",
                options=range(len(suggested_intervals)),
                format_func=lambda i: f"[{suggested_intervals[i][0]}, {suggested_intervals[i][1]}]"
            )
            bisection_settings["a"] = suggested_intervals[selected_interval][0]
            bisection_settings["b"] = suggested_intervals[selected_interval][1]
            
            with st.expander("Advanced Settings"):
                bisection_settings["delta_exp"] = st.slider(
                    "Step size tolerance (10^x)",
                    min_value=-15, max_value=-5, value=-10,
                    help="Tolerance for interval width"
                )
                bisection_settings["epsilon_exp"] = st.slider(
                    "Function value tolerance (10^x)",
                    min_value=-15, max_value=-5, value=-10,
                    help="Tolerance for function value"
                )
                bisection_settings["max_iterations"] = st.slider(
                    "Maximum iterations",
                    min_value=10, max_value=200, value=100,
                    help="Maximum number of iterations to perform"
                )
        
        bisection_settings["use"] = use_bisection
        
        # Newton's Method settings
        st.subheader("Newton's Method")
        use_newton = st.checkbox("Use Newton's Method", value=True)
        
        newton_settings = {}
        if use_newton:
            newton_settings["x0"] = st.number_input(
                "Initial guess",
                value=function_details["suggested_intervals"][0][0],
                help="Starting point for Newton's method"
            )
            
            with st.expander("Advanced Settings"):
                newton_settings["delta1_exp"] = st.slider(
                    "Step size tolerance (10^x)",
                    min_value=-15, max_value=-5, value=-10,
                    key="newton_delta1",
                    help="Tolerance for step size"
                )
                newton_settings["delta2_exp"] = st.slider(
                    "Function value tolerance (10^x)",
                    min_value=-15, max_value=-5, value=-10,
                    key="newton_delta2",
                    help="Tolerance for function value"
                )
                newton_settings["epsilon_exp"] = st.slider(
                    "Derivative tolerance (10^x)",
                    min_value=-15, max_value=-5, value=-10,
                    key="newton_epsilon",
                    help="Tolerance to avoid division by near-zero derivative"
                )
                newton_settings["max_iterations"] = st.slider(
                    "Maximum iterations",
                    min_value=10, max_value=200, value=100,
                    key="newton_max_iter",
                    help="Maximum number of iterations to perform"
                )
        
        newton_settings["use"] = use_newton
        
        # Secant Method settings
        st.subheader("Secant Method")
        use_secant = st.checkbox("Use Secant Method", value=True)
        
        secant_settings = {}
        if use_secant:
            secant_settings["a"] = st.number_input(
                "First initial guess",
                value=function_details["suggested_intervals"][0][0],
                key="secant_a",
                help="First point for secant method"
            )
            secant_settings["b"] = st.number_input(
                "Second initial guess",
                value=function_details["suggested_intervals"][0][1],
                key="secant_b",
                help="Second point for secant method"
            )
            
            with st.expander("Advanced Settings"):
                secant_settings["delta1_exp"] = st.slider(
                    "Step size tolerance (10^x)",
                    min_value=-15, max_value=-5, value=-10,
                    key="secant_delta1",
                    help="Tolerance for step size"
                )
                secant_settings["delta2_exp"] = st.slider(
                    "Function value tolerance (10^x)",
                    min_value=-15, max_value=-5, value=-10,
                    key="secant_delta2",
                    help="Tolerance for function value"
                )
                secant_settings["max_iterations"] = st.slider(
                    "Maximum iterations",
                    min_value=10, max_value=200, value=100,
                    key="secant_max_iter",
                    help="Maximum number of iterations to perform"
                )
        
        secant_settings["use"] = use_secant
        
        # Collect all settings
        settings = {
            "bisection": bisection_settings,
            "newton": newton_settings,
            "secant": secant_settings
        }
        
        # Run button
        if st.button("Run Analysis", type="primary"):
            st.session_state.results = run_root_finding_methods(selected_function, settings)
            st.session_state.selected_function = selected_function
            st.session_state.settings = settings
    
    # Main content area
    st.title("Root-Finding Methods Analysis")
    st.write("""
    This application implements and analyzes three numerical methods for finding roots of nonlinear equations:
    the **Bisection Method**, **Newton's Method**, and the **Secant Method**.
    """)
    
    # Create tabs for different views
    tabs = st.tabs(["Analysis Dashboard", "Detailed Report", "Pseudocode", "Code Report"])
    
    # Check if results are available
    has_results = "results" in st.session_state
    
    # Analysis Dashboard Tab
    with tabs[0]:
        if not has_results:
            st.info("Configure the settings in the sidebar and click 'Run Analysis' to get started.")
        else:
            results = st.session_state.results
            selected_function = st.session_state.selected_function
            func_details = get_function_details(selected_function)
            f = func_details["function"]
            
            st.header(f"Analysis of {func_details['display_name']}")
            
            # Function plot with roots
            st.subheader("Function Visualization")
            
            # Determine suitable x range for visualization
            x_range = func_details["suggested_intervals"][0]
            x_range = (x_range[0] - 1, x_range[1] + 1)  # Extend a bit for better visualization
            
            # Find all roots from results
            roots = []
            for method_name, result in results.items():
                if result.get('converged', False) and result.get('root') is not None:
                    roots.append((method_name, result['root']))
            
            if roots:
                fig = plot_function(
                    f, x_range,
                    title=f"Plot of {func_details['display_name']}",
                    root=roots[0][1] if roots else None
                )
                st.pyplot(fig)
                plt.close(fig)
            
            # Results summary
            st.subheader("Summary of Results")
            
            # Create a result table
            result_data = []
            for method_name, result in results.items():
                result_data.append({
                    "Method": method_name,
                    "Root Found": f"{result.get('root', 'N/A'):.10f}" if result.get('root') is not None else "N/A",
                    "Iterations": result.get('iterations_count', 'N/A'),
                    "Converged": "✅" if result.get('converged', False) else "❌",
                    "Final Error": f"{result.get('error_history', [0])[-1]:.10e}" if result.get('error_history') else "N/A",
                    "Est. Convergence Rate": f"{result.get('convergence_rate', 'N/A'):.4f}" if result.get('convergence_rate') is not None else "N/A"
                })
            
            # Display result table
            st.table(result_data)
            
            # Comparison of convergence rates
            st.subheader("Convergence Comparison")
            
            # Check if we have at least two methods with results
            if len(results) >= 2:
                fig = compare_convergence_rates(
                    results,
                    title=f"Convergence Rate Comparison for {func_details['display_name']}"
                )
                st.pyplot(fig)
                plt.close(fig)
                
                st.write("""
                The graph above shows the convergence behavior of each method on a logarithmic scale.
                Steeper slopes indicate faster convergence. Newton's method typically shows the steepest
                decline, followed by secant, and then bisection.
                """)
            else:
                st.info("Run at least two methods to see a convergence comparison.")
            
            # Detailed method results
            st.header("Detailed Method Results")
            
            # Create tabs for each method
            method_tabs = st.tabs(list(results.keys()))
            
            for i, (method_name, result) in enumerate(results.items()):
                with method_tabs[i]:
                    if not result.get('converged', False):
                        st.error(f"Method did not converge. Reason: {result.get('error_message', 'Unknown')}")
                    
                    # Method result summary
                    root_value = result.get('root')
                    conv_rate = result.get('convergence_rate')
                    
                    st.markdown(f"""
                    **Root found**: {f"{root_value:.10f}" if root_value is not None else "N/A"}  
                    **Iterations**: {result.get('iterations_count', 'N/A')}  
                    **Convergence rate**: {f"{conv_rate:.4f}" if conv_rate is not None else "N/A"}
                    """)
                    
                    # Error convergence plot
                    if result.get('error_history'):
                        st.subheader("Error Convergence")
                        fig = plot_error_convergence(
                            result['error_history'],
                            title=f"Error Convergence for {method_name}",
                            method_name=method_name,
                            rate=result.get('convergence_rate')
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Function visualization with iterations
                    st.subheader("Iteration Visualization")
                    
                    # Handle method-specific visualization
                    method_type = method_name.split()[0].lower()  # Extract first word (bisection, newton, secant)
                    
                    # Prepare iterations data
                    iterations = result.get('iterations', [])
                    if method_type and iterations:
                        # Show current iteration with a slider
                        max_iter = len(iterations) - 1
                        if max_iter > 0:
                            iter_slider = st.slider(
                                "Iteration",
                                min_value=0,
                                max_value=max_iter,
                                value=min(5, max_iter),
                                key=f"{method_name}_slider"
                            )
                            
                            # Generate animation frames
                            animation_frames = plot_function_with_iterations_animation(
                                f, iterations, method_type, x_range
                            )
                            
                            # Display selected frame
                            if animation_frames and 0 <= iter_slider < len(animation_frames):
                                st.pyplot(animation_frames[iter_slider])
                                plt.close(animation_frames[iter_slider])
                    
                    # Iteration table
                    st.subheader("Iteration Details")
                    if iterations:
                        df = create_iteration_table(iterations, method_type)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No iteration data available.")
    
    # Detailed Report Tab
    with tabs[1]:
        if has_results:
            render_report(selected_function, st.session_state.results)
        else:
            st.info("Run the analysis first to generate a detailed report.")
    
    # Pseudocode Tab
    with tabs[2]:
        render_pseudocode()
    
    # Code Report Tab
    with tabs[3]:
        def render_code_report():
            """
            Render the Code Report tab content with detailed information about the project implementation.
            """
            st.markdown("# Code Report")
            
            st.markdown("""
            ## Developer Information
            
            **Developer:** John Akujobi  
            **GitHub:** [github.com/jakujobi](https://github.com/jakujobi)  
            **Website:** [jakujobi.com](https://jakujobi.com)  
            **Project:** Project 2 for Math 374: Scientific Computation (Spring 2025)  
            **Institution:** South Dakota State University  
            **Professor:** Dr. Jung-Han Kimn  
            
            ## Project Architecture
            
            The project follows a modular architecture to promote code reusability, maintainability, 
            and separation of concerns. It is organized into the following modules:
            
            ### Project Structure
            
            ```
            Math374P2/
            ├── Modules/
            │   ├── numerical_methods.py  # Implementation of root-finding methods
            │   ├── test_functions.py     # Test functions and their derivatives
            │   ├── visualization.py      # Functions for plotting and data visualization
            │   └── report.py             # Report generation and mathematical explanations
            ├── streamlit_app.py          # Main Streamlit application
            ├── requirements.txt          # Project dependencies
            └── README.md                 # Project documentation
            ```
            
            ### 1. Numerical Methods (`numerical_methods.py`)
            
            This module implements the core numerical algorithms for root finding.
            
            #### Key Functions:
            
            1. **`bisection_method(f, a, b, delta, epsilon, max_iterations)`**
                - **Purpose**: Implements the bisection algorithm, which repeatedly bisects an interval and selects the subinterval containing the root.
                - **Implementation Details**:
                    - Uses the Intermediate Value Theorem: if f(a) and f(b) have opposite signs, there must be a root in [a, b]
                    - At each iteration, computes the midpoint c = (a + b)/2 and determines which subinterval contains the root
                    - Tracks the error history and function values at each iteration
                    - Applies termination criteria based on interval width and function value tolerances
            
            2. **`newton_method(f, df, x0, delta1, delta2, epsilon, max_iterations)`**
               - **Purpose**: Implements Newton's method, which uses function derivatives to quickly converge to a root.
               - **Implementation Details**:
                    - Uses the formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
                    - Checks for small derivatives to avoid numerical instability
                    - Tracks the error history and convergence behavior
                    - Applies termination criteria based on step size and function value tolerances
            
            3. **`secant_method(f, a, b, delta1, delta2, max_iterations)`**
               - **Purpose**: Implements the secant method, which approximates the derivative using finite differences.
               - **Implementation Details**:
                    - Uses the formula: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1})/(f(x_n) - f(x_{n-1}))
                    - Checks for near-zero denominators to ensure numerical stability
                    - Swaps points based on function value magnitudes for better convergence
                    - Tracks iterations and error history
            
            4. **`estimate_convergence_rate(error_history)`**
               - **Purpose**: Estimates the convergence rate of a numerical method based on the error history.
               - **Implementation Details**:
                    - Uses linear regression on the log-log plot of consecutive errors
                    - The slope of this line approximates the order of convergence
            """)
            
            st.markdown("""
            ### 2. Test Functions (`test_functions.py`)
            
            This module defines the test functions and their derivatives used to evaluate the numerical methods.
            
            #### Key Functions:
            
            1. **Test Functions**:
                - `f1(x) = x^2 - 4*sin(x)` - Combines polynomial and trigonometric terms
                - `f2(x) = x^2 - 1` - Simple quadratic with known roots at ±1
                - `f3(x) = x^3 - 3*x^2 + 3*x - 1` - Cubic polynomial with a triple root at x=1
            
            2. **Derivative Functions**:
                - `df1(x) = 2*x - 4*cos(x)`
                - `df2(x) = 2*x`
                - `df3(x) = 3*x^2 - 6*x + 3`
            
            3. **Utility Functions**:
                - `get_function_details(function_id)` - Retrieves information about a specific test function
                - `get_all_functions()` - Returns a dictionary of all available test functions
            
            ### 3. Visualization (`visualization.py`)
            
            This module provides functions for visualizing the numerical methods, their convergence, and comparisons.
            
            #### Key Functions:
            
            1. **`plot_function(f, x_range, title, ...)`**
                - Plots a function over a specified range with optional root marking
            
            2. **`plot_error_convergence(error_history, ...)`**
                - Visualizes how the error decreases with iterations, using logarithmic scaling
            
            3. **`compare_convergence_rates(results, ...)`**
                - Creates a comparative visualization of convergence rates across different methods
            
            4. **`create_iteration_table(iterations, method)`**
                - Generates a formatted table of iteration details suitable for display
            
            5. **`plot_function_with_iterations_animation(f, iterations, method, ...)`**
                - Creates an animated visualization showing how each method progresses toward the root
            """)
            
            st.markdown("""
            ### 4. Report Generation (`report.py`)
            
            This module provides the content and structure for the detailed mathematical report.
            
            #### Key Functions:
            
            1. **`render_introduction()`** - Introduces the root-finding methods
            2. **`render_methods_explanation()`** - Explains the algorithms and pseudocode
            3. **`render_termination_criteria()`** - Details the convergence conditions
            4. **`render_convergence_rates()`** - Explains the theoretical and practical convergence rates
            5. **`render_method_comparison(results)`** - Compares the methods' performance
            6. **`render_function_analysis(function_id)`** - Analyzes specific test functions
            7. **`render_conclusion(results)`** - Summarizes findings and insights
            8. **`render_report(active_function, results)`** - Renders the complete report
            9. **`render_pseudocode()`** - Displays detailed pseudocode for all three methods
            
            ### 5. Main Application (`streamlit_app.py`)
            
            This module integrates all other modules to create an interactive web application.
            
            #### Key Functions:
            
            1. **`run_root_finding_methods(function_id, settings)`**
                - Runs all three numerical methods with the specified settings
                - Collects and processes results for visualization and analysis
            
            2. **`main()`**
                - Sets up the Streamlit interface with sidebar controls
                - Creates tabs for different views (Analysis, Report, Pseudocode, Code Report)
                - Handles user interaction and visualization rendering
            """)
            
            st.markdown("""
            ## Implementation Details
            
            ### Data Structures
            
            1. **Method Results**: Each numerical method returns a dictionary containing:
                - `root`: The approximated root value
                - `iterations`: Detailed information about each iteration
                - `converged`: Boolean indicating successful convergence
                - `iterations_count`: Total number of iterations performed
                - `error_history`: List of error values at each iteration
                - `function_values`: List of function values at each iteration
            
            2. **Function Details**: Each test function is represented by a dictionary with:
                - `function`: The actual function object
                - `derivative`: The derivative function object
                - `display_name`: User-friendly name for display
                - `latex`: LaTeX representation for mathematical display
                - `description`: Brief description of the function
                - `suggested_intervals`: Recommended intervals for root finding
            
            ### Design Patterns and Best Practices
            
            1. **Modular Design**: The codebase is organized into logical modules with clear responsibilities
            
            2. **Type Annotations**: Python type hints are used throughout the codebase to improve code clarity and enable static type checking
            
            3. **Comprehensive Documentation**: All functions have docstrings that explain their purpose, parameters, and return values
            
            4. **Error Handling**: Robust error handling is implemented, particularly for potential numerical instabilities
            
            5. **Configurability**: Method parameters are configurable through the UI, allowing for experimentation
            
            ## Development Process and Challenges
            
            ### Development Process
            
            The application was developed following these steps:
            
            1. **Requirements Analysis**: Understanding the mathematical theory and requirements
            2. **Module Design**: Planning the modular structure of the application
            3. **Implementation**: Coding the numerical methods with careful attention to numerical stability
            4. **Visualization**: Creating informative visualizations of the methods and their results
            5. **UI Development**: Building an intuitive Streamlit interface for user interaction
            6. **Testing**: Verifying correct behavior on various test functions
            7. **Documentation**: Adding comprehensive documentation throughout the codebase
            8. **Refinement**: Improving the UI and fixing any issues
            
            ### Challenges and Solutions
            
            1. **Numerical Stability**: 
                - Challenge: Methods like Newton's and secant can encounter division by near-zero values
                - Solution: Implemented tolerance checks and safeguards against numerical instability
            
            2. **Convergence Rate Estimation**: 
                - Challenge: Accurately estimating the order of convergence from empirical data
                - Solution: Used linear regression on log-log plots of consecutive errors
            
            3. **Visualization Complexity**: 
                - Challenge: Creating intuitive visualizations of iteration processes
                - Solution: Developed animated visualizations that show the progression toward roots
            
            4. **User Interface Design**: 
                - Challenge: Balancing complexity and usability in the interface
                - Solution: Organized the interface into tabs and used expandable sections for advanced settings
            """)
        
        render_code_report()

if __name__ == "__main__":
    main()
