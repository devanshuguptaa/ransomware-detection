import streamlit as st
from utils.helpers import get_uploaded_or_default

st.set_page_config(
    page_title="Ransomware Detection - Home",
    layout="wide",
    page_icon="🛡️"
)

# Sidebar layout
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/5610/5610944.png", width=100)
st.sidebar.title("🔐 Ransomware Detection App")

st.sidebar.info("Navigate through the sections using the sidebar menu.")

# Upload CSV file
st.sidebar.markdown("### 📂 Upload Ransomware CSV")
uploaded_file = st.sidebar.file_uploader("Upload your `Ransomware.csv` file", type=["csv"])

# Load uploaded or default
df = get_uploaded_or_default(uploaded_file)

# Don't show anything until file is uploaded or fallback is loaded
if df is not None:

    # Title and intro
    st.title("🛡️ Ransomware Detection using Machine Learning")
    st.markdown("---")

    st.markdown("""
    ### 🎯 **Problem Statement**
    > Ransomware is a type of malicious software designed to block access to files or systems until a ransom is paid.

    This project focuses on:
    - Detecting ransomware from file metadata using Machine Learning
    - Using **LIME** to explain model predictions
    - Fixing data imbalance with **SMOTE + Tomek Links**
    """)
         
    # Project Overview
    st.markdown("---")
    st.markdown("### 📚 **Project Overview**")
    st.markdown("""
    - **Section 1:** Data Exploration and Preprocessing  
    - **Section 2:** Visual Analysis & Correlation  
    - **Section 3:** Feature Selection using IV & WoE  
    - **Section 4:** Model Training and Evaluation  
    - **Section 5:** Model Explainability with **LIME**  
    - **Section 6:** Data Balancing with **SMOTE Tomek**  
    """)
          
    # Footer
    st.markdown("---")
    st.success("🚀 Let's build an interpretable and effective ML model to detect ransomware attacks!")
    