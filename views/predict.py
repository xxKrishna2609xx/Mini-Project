import streamlit as st
import time
from utils.ml_logic import predict_job, highlight_suspicious_keywords, is_valid_english_job_text, get_shap_explanations

def render_predict(models, vectorizer):
    st.markdown("<h2 class='sub-header'>🔍 Authenticity Checker</h2>", unsafe_allow_html=True)
    
    # Model Selection UI
    st.markdown("### 🧠 Model Selection")
    col_mod_1, col_mod_2 = st.columns([1, 2])
    with col_mod_1:
        model_name = st.selectbox("Choose ML Model", list(models.keys()), index=0)
        selected_model = models[model_name]
    with col_mod_2:
        st.write("") # spacing
        st.write("") # spacing
        compare_all = st.toggle("Compare All Models side-by-side", help="Run predictions across all available models.")
        
    model_descriptions = {
        "Random Forest": "🌟 **Ensemble model** providing high accuracy and balanced precision/recall, minimizing false positives.",
        "Logistic Regression": "⚡ **Fast and interpretable** linear classifier, excellent as a robust baseline.",
        "Naive Bayes": "📊 **Probabilistic model** (MultinomialNB) highly effective for text classification and spam detection."
    }
    st.info(model_descriptions.get(model_name, ""))
    
    st.markdown("---")

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
                        
                    # Dictionary Validation / Gibberish Check
                    is_valid, invalid_reason = is_valid_english_job_text(job['text'])
                    if not is_valid:
                        company_str = f" at {job['company']}" if job['company'] and job['company'] != "Uploaded File" else ""
                        st.markdown(f"#### {job['title']}{company_str}")
                        st.error(f"🚫 **INVALID TEXT DETECTED:** {invalid_reason} The machine learning models cannot accurately assess gibberish or heavily malformed input strings.")
                        st.markdown("<br>", unsafe_allow_html=True)
                        continue
                        
                    # Execute model analysis block
                    company_str = f" at {job['company']}" if job['company'] and job['company'] != "Uploaded File" else ""
                    st.markdown(f"#### {job['title']}{company_str}")
                    
                    highlighted_diff, flags, top_tfidf = highlight_suspicious_keywords(job['text'], vectorizer, top_n=5)
                    
                    models_to_run = models if compare_all else {model_name: selected_model}
                    
                    cols = st.columns(len(models_to_run))
                    
                    for idx, (m_name, m_obj) in enumerate(models_to_run.items()):
                        is_scam, confidence = predict_job(job['text'], m_obj, vectorizer)
                        
                        with cols[idx]:
                            if is_scam:
                                st.markdown(
                                    f"""
                                    <div class='result-card-scam' style='padding: 15px; border-radius: 8px;'>
                                        <div style='font-size:0.8rem; opacity: 0.7; color: white;'>Model: {m_name}</div>
                                        <h3 style='margin-top: 5px; color: white;'>🚨 SCAM</h3>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div class='result-card-genuine' style='padding: 15px; border-radius: 8px;'>
                                        <div style='font-size:0.8rem; opacity: 0.7; color: white;'>Model: {m_name}</div>
                                        <h3 style='margin-top: 5px; color: white;'>✅ GENUINE</h3>
                                    </div>
                                    """, unsafe_allow_html=True
                                )
                                
                            st.write(f"**Confidence:** {confidence:.2%}")
                            st.progress(float(confidence))
                            
                    st.markdown("##### 🧠 Explainability")
                    col_expl_1, col_expl_2 = st.columns(2)
                    with col_expl_1:
                        if flags:
                            st.warning(f"🚩 **Top Suspicious Keywords Detected:** {', '.join(flags)}")
                        else:
                            st.info("No common suspicious keywords detected.")
                    with col_expl_2:
                        if top_tfidf:
                            st.success(f"🔑 **Most Influential Words (TF-IDF):** {', '.join(top_tfidf)}")
                        else:
                            st.info("Insufficient TF-IDF influencers extracted.")

                    if flags or top_tfidf:
                        with st.expander(f"View Keyword Highlighting for {job['title']}"):
                            st.markdown(highlighted_diff, unsafe_allow_html=True)
                            
                    st.markdown("<hr style='margin: 15px 0; opacity: 0.2;'>", unsafe_allow_html=True)
                    st.markdown(f"##### 🧠 AI Explanation (Model-Based approach via {model_name})")
                    
                    top_scam, top_genuine = get_shap_explanations(job['text'], selected_model, model_name, vectorizer)
                    
                    if top_scam or top_genuine:
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            if top_scam:
                                scam_strs = [f"- **\"{w}\"** → strong scam signal" for w,v in top_scam]
                                st.error("⚠️ **Influential Scam Drivers:**\n\n" + "\n".join(scam_strs))
                            else:
                                st.info("No strong scam drivers found.")
                        with col_s2:
                            if top_genuine:
                                gen_strs = [f"- **\"{w}\"** → genuine signal" for w,v in top_genuine]
                                st.success("✅ **Influential Genuine Drivers:**\n\n" + "\n".join(gen_strs))
                            else:
                                st.info("No strong genuine drivers found.")
                    else:
                        st.info(f"Native model-based SHAP explanations are unsupported for {model_name}.")
                        
                    st.markdown("<br>", unsafe_allow_html=True)
