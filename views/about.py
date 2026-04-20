import streamlit as st

def render_about():
    st.markdown("<h2 class='sub-header'>📘 About the Project</h2>", unsafe_allow_html=True)
    
    st.markdown(
        """
        ### 🚨 The Problem
        Employment scams are on the rise. Fraudsters create fake job listings to steal personal information, 
        bank details, or solicit upfront payments from a vulnerable job-seeking population. It is increasingly 
        difficult to distinguish between genuine opportunities and malicious scams.
        
        ### 💡 Our Solution
        This application uses Natural Language Processing (NLP) and Machine Learning techniques to automatically 
        analyze job posting descriptions and compute a risk score. By identifying specific linguistic patterns and 
        suspicious keywords, we can alert users before they fall victim.
        
        ### ⚙️ Technologies Used
        * **Frontend**: Streamlit, HTML/CSS injected via Markdown
        * **Backend Data Manipulation**: Pandas, NumPy
        * **Machine Learning & NLP**: Scikit-Learn (TF-IDF Vectorizer, Random Forest Classifier)
        * **Data Visualization**: Plotly
        
        ### 🚀 Real-World Impact
        Automated scam detection systems can be integrated into large job boards (e.g., LinkedIn, Indeed) 
        to flag and remove fraudulent posts before they reach the applicant, creating a safer ecosystem for everyone.
        """
    )
    
    st.info("Developed for Academic & Professional Demonstration.")
