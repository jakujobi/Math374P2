# Root-Finding Methods Analysis

## MATH 374 Project 2: Scientific Computation (Spring 2025)

This project implements and analyzes three numerical methods for finding roots of nonlinear equations:

- Bisection Method
- Newton's Method
- Secant Method

The application tests these methods on three specific test functions:

- f₁(x) = x² - 4sin(x)
- f₂(x) = x² - 1
- f₃(x) = x³ - 3x² + 3x - 1

## Live Demo

A live version of this project is hosted online at: [https://math374p2.streamlit.app/](https://math374p2.streamlit.app/)

## Features

- **Interactive UI**: Select test functions and configure method parameters
- **Comprehensive Visualization**: View function plots, iteration steps, and error convergence
- **Comparative Analysis**: Compare convergence rates across methods
- **Detailed Report**: Access in-depth mathematical explanations of each method
- **Pseudocode Documentation**: Study the algorithms behind each method
- **Code Report**: Review implementation details and project architecture

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

   ```bash
   git clone https://github.com/jakujobi/Math374P2.git
   cd Math374P2
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the application locally:

```bash
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
7. **Code Report**: Explore the implementation details and architecture in the "Code Report" tab

## Integrated Documentation

The project report is integrated directly into the Streamlit application using Markdown text. This approach was chosen for several reasons:

- **Efficiency**: Streamlines the development and documentation process
- **Change Tracking**: Makes it easier to track changes to both code and documentation
- **Direct References**: Allows for direct references to application components
- **Unified Access**: Enables users to access and review all project information from a single location

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

## References

- Cheney, W., & Kincaid, D. (2012). Numerical Mathematics and Computing (7th ed.).
- Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.).
- Atkinson, K. E. (1989). An Introduction to Numerical Analysis (2nd ed.).

## Developer Information

- **Developer**: John Akujobi
- **GitHub**: [github.com/jakujobi](https://github.com/jakujobi)
- **Website**: [jakujobi.com](https://jakujobi.com)
- **Institution**: South Dakota State University
- **Professor**: Dr. Jung-Han Kimn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
