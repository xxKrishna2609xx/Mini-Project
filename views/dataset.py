import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    if os.path.exists("fake_job_postings.csv"):
        return pd.read_csv("fake_job_postings.csv")
    return None

def render_dataset():
    st.markdown("<h2 class='sub-header'>📂 Dataset Preview</h2>", unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        st.write("Browse the dataset used to train the machine learning models. Labels: **1 = Scam**, **0 = Genuine**.")
        
        # Display dataset statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            scam_count = len(df[df['fraudulent'] == 1])
            st.metric("Scam Postings", scam_count)
        with col3:
            genuine_count = len(df[df['fraudulent'] == 0])
            st.metric("Genuine Postings", genuine_count)
            
        st.markdown("### Interactive Data Table")
        st.dataframe(df, use_container_width=True)
        
    else:
        st.error("Dataset not found. Please run `setup_mock_model.py` to generate the mock dataset.")
