# Root-Finding Methods Analysis

## MATH 373 Project 2: Numerical Methods for Root Finding

This project implements and analyzes three numerical methods for finding roots of nonlinear equations:
- Bisection Method
- Newton's Method
- Secant Method

The application tests these methods on three specific test functions:
- f₁(x) = x² - 4sin(x)
- f₂(x) = x² - 1
- f₃(x) = x³ - 3x² + 3x - 1

## Features

- **Interactive UI**: Select test functions and configure method parameters
- **Comprehensive Visualization**: View function plots, iteration steps, and error convergence
- **Comparative Analysis**: Compare convergence rates across methods
- **Detailed Report**: Access in-depth mathematical explanations of each method
- **Pseudocode Documentation**: Study the algorithms behind each method

## Project Structure

```
Math374P2/
├── Modules/
│   ├── numerical_methods.py  # Implementation of the three root-finding methods
│   ├── test_functions.py     # Test functions and their derivatives
│   ├── visualization.py      # Visualization functions for plotting
│   └── report.py             # Report generation and mathematical explanations
├── streamlit_app.py          # Main Streamlit application entry point
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Math374P2.git
   cd Math374P2
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the application locally:

```
streamlit run streamlit_app.py
```

This will start a local web server and automatically open the application in your default web browser.

## Usage Guide

1. **Select a Test Function**: Choose from the three predefined functions in the sidebar
2. **Configure Method Settings**: Set parameters for each method (initial guesses, tolerances)
3. **Run Analysis**: Click the "Run Analysis" button to compute roots and visualize results
4. **Explore Results**: 
   - View function plots with marked roots
   - Examine convergence behavior through graphs
   - Study iteration details in tables
   - Compare performance across methods
5. **Read the Report**: Access the "Detailed Report" tab for mathematical explanations
6. **Review Pseudocode**: Examine the algorithmic details in the "Pseudocode" tab

## Key Questions Addressed

1. **Termination Criteria**: The application uses both step size and function value tolerances to determine convergence.
2. **Convergence Rates**: For each method, the convergence rate is estimated and displayed.
3. **Comparative Analysis**: Results are compared across methods, highlighting trade-offs between speed and reliability.

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Pandas**: Data handling for tables

## Author

[Your Name]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
