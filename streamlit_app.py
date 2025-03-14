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
    page_title="Root-Finding Methods Analysis",
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
</style>
""", unsafe_allow_html=True)

def run_root_finding_methods(function_id, settings):
    """
    Run all three root-finding methods on the selected function with specified settings.
    
    Args:
        function_id: ID of the function to analyze
        settings: Dictionary of settings for each method
        
    Returns:
        Dictionary with results from each method
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
    tabs = st.tabs(["Analysis Dashboard", "Detailed Report", "Pseudocode"])
    
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
                    st.markdown(f"""
                    **Root found**: {result.get('root', 'N/A'):.10f}  
                    **Iterations**: {result.get('iterations_count', 'N/A')}  
                    **Convergence rate**: {result.get('convergence_rate', 'N/A'):.4f if result.get('convergence_rate') is not None else "N/A"}
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


if __name__ == "__main__":
    main()
