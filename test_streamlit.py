import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Test App",
    page_icon="📊",
    layout="wide",
)

st.title("Test Streamlit App")
st.write("This is a test to verify that Streamlit is working correctly.")

# Add footer
st.markdown(
    """
    <style>
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
    </style>
    <div class="footer">
        Project 2 for Math 374: Scientific Computation (Spring 2025) | 
        South Dakota State University | 
        Developed by John Akujobi
    </div>
    """,
    unsafe_allow_html=True,
)
