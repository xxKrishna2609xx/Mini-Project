import streamlit as st
import os

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Import Views and Utilities
from views.predict import render_predict
from views.insights import render_insights
from views.dataset import render_dataset
from views.about import render_about
from utils.ml_logic import load_models

# Load Models globally
model, vectorizer = load_models()

# Main Header
st.markdown("<h1 class='main-header'>🛡️ Fake Job & Internship Scam Detection System</h1>", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3248/3248060.png", width=80) 
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["🏠 Home (Prediction)", "📊 Model Insights", "📂 Dataset Preview", "📘 About Project"]
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    if model and vectorizer:
        st.success("✅ Models Loaded")
    else:
        st.error("❌ Models Missing")
        st.caption("Please run `setup_mock_model.py` first.")

# Page Routing
if page == "🏠 Home (Prediction)":
    render_predict(model, vectorizer)
elif page == "📊 Model Insights":
    render_insights()
elif page == "📂 Dataset Preview":
    render_dataset()
elif page == "📘 About Project":
    render_about()

# Footer
st.markdown("<div class='footer'>Developed for Academic Excellence | ML & NLP Powered</div>", unsafe_allow_html=True)
