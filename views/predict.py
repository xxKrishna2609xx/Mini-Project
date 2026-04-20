import streamlit as st
import time
from utils.ml_logic import predict_job, highlight_suspicious_keywords

def render_predict(model, vectorizer):
    st.markdown("<h2 class='sub-header'>🔍 Authenticity Checker</h2>", unsafe_allow_html=True)
    
    # Initialize session state for multiple entries
    if "manual_entry_count" not in st.session_state:
        st.session_state.manual_entry_count = 1

    # Input Option Radio
    input_type = st.radio("Choose Input Method", ("Manual Entry", "File Upload"), horizontal=True)
    
    jobs_to_check = []

    if input_type == "Manual Entry":
        for i in range(st.session_state.manual_entry_count):
            if st.session_state.manual_entry_count > 1:
                st.markdown(f"#### Job Entry {i + 1}")
                
            col1, col2 = st.columns([1, 1])
            with col1:
                title = st.text_input("Job Title", placeholder="e.g., Software Engineer", key=f"title_{i}")
            with col2:
                company = st.text_input("Company Name (Optional)", placeholder="e.g., Tech Corp", key=f"company_{i}")
                
            desc = st.text_area("Job Description", height=150, placeholder="Paste the job description here...", key=f"desc_{i}")
            reqs = st.text_area("Requirements / Skills (Optional)", height=100, placeholder="Paste skills required...", key=f"reqs_{i}")
            
            job_text = f"{title} {desc} {reqs}".strip()
            if job_text:
                jobs_to_check.append({
                    "id": i + 1,
                    "title": title if title else f"Entry {i+1}",
                    "company": company,
                    "text": job_text
                })
            
            if i < st.session_state.manual_entry_count - 1:
                st.markdown("---")
                
        # Add / Remove Buttons
        col_btn1, col_btn2, _ = st.columns([1, 1, 3])
        with col_btn1:
            if st.button("➕ Add Another Job"):
                st.session_state.manual_entry_count += 1
                st.rerun()
        with col_btn2:
            if st.session_state.manual_entry_count > 1:
                if st.button("➖ Remove Last Job"):
                    st.session_state.manual_entry_count -= 1
                    st.rerun()
                    
        st.markdown("<br>", unsafe_allow_html=True)
        
        check_btn_text = "Check Job Authenticity" if st.session_state.manual_entry_count == 1 else "Check All Jobs"
        submit_btn = st.button(check_btn_text, type="primary")

    else:
        uploaded_files = st.file_uploader("Upload Job Descriptions (.txt)", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                job_text = str(uploaded_file.read(), "utf-8").strip()
                jobs_to_check.append({
                    "id": i + 1,
                    "title": uploaded_file.name,
                    "company": "Uploaded File",
                    "text": job_text
                })
                
            with st.expander(f"Preview Uploaded Texts ({len(uploaded_files)} files)"):
                for job in jobs_to_check:
                    st.markdown(f"**{job['title']}**")
                    st.markdown(f"<div style='max-height: 100px; overflow-y: auto; font-size: 0.9em; color: #a0aab2;'>{job['text']}</div>", unsafe_allow_html=True)
                    st.markdown("---")
                    
        submit_btn = st.button("Check Job Authenticity", type="primary")

    # Evaluate all collected jobs
    if submit_btn:
        if not jobs_to_check:
            st.warning("Please provide at least one valid job description.")
        else:
            with st.spinner(f"Analyzing {len(jobs_to_check)} job{'s' if len(jobs_to_check) > 1 else ''}..."):
                time.sleep(1) # Fake loading for effect
                
                st.markdown("---")
                st.subheader(f"Results ({len(jobs_to_check)} Entries Checked)")
                
                for job in jobs_to_check:
                    if len(job['text']) < 10:
                        st.warning(f"**{job['title']}**: Text too short (at least 10 characters required) to analyze.")
                        continue
                        
                    is_scam, confidence = predict_job(job['text'], model, vectorizer)
                    
                    if is_scam is not None:
                        company_str = f" at {job['company']}" if job['company'] and job['company'] != "Uploaded File" else ""
                        st.markdown(f"#### {job['title']}{company_str}")
                        
                        # Highlight Text
                        highlighted_diff, flags = highlight_suspicious_keywords(job['text'])
                        
                        if is_scam:
                            st.markdown(
                                f"""
                                <div class='result-card-scam' style='padding: 15px; border-radius: 8px;'>
                                    <h3 style='margin-top: 0;'>🚨 SCAM DETECTED</h3>
                                    <p style='margin-bottom: 0;'>This job posting exhibits characteristics commonly associated with fraudulent offers.</p>
                                </div>
                                """, unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class='result-card-genuine' style='padding: 15px; border-radius: 8px;'>
                                    <h3 style='margin-top: 0;'>✅ GENUINE JOB</h3>
                                    <p style='margin-bottom: 0;'>This job posting looks legitimate based on our model's criteria.</p>
                                </div>
                                """, unsafe_allow_html=True
                            )
                        
                        # Metrics & Confidence
                        st.write(f"**Model Confidence:** {confidence:.2%}")
                        st.progress(float(confidence))
                        
                        # Flag findings
                        if flags:
                            st.warning(f"🚩 **Suspicious Keywords Found:** {', '.join(flags)}")
                            with st.expander(f"View Highlighted Text for {job['title']}"):
                                st.markdown(highlighted_diff, unsafe_allow_html=True)
                        else:
                            st.info("No common suspicious keywords detected in the text structure.")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
