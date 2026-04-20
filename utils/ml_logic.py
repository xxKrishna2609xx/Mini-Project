import streamlit as st
import joblib
import os
import re
import numpy as np
import shap

@st.cache_resource
def load_models():
    """Loads and caches the trained models and vectorizer."""
    required_files = ["rf_model.pkl", "lr_model.pkl", "nb_model.pkl", "vectorizer.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"Model file {file} not found. Please run `setup_mock_model.py` to generate them.")
            return None, None
            
    try:
        models = {
            "Random Forest": joblib.load("rf_model.pkl"),
            "Logistic Regression": joblib.load("lr_model.pkl"),
            "Naive Bayes": joblib.load("nb_model.pkl")
        }
        vectorizer = joblib.load("vectorizer.pkl")
        return models, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def is_valid_english_job_text(text):
    """Simple heuristic validation to detect pure gibberish and keyboard smashes."""
    if len(text) < 10:
        return True # Handled elsewhere
        
    # Check 1: Vowel to consonant ratio. Keyboard smashes often lack vowels.
    vowels = sum(1 for char in text.lower() if char in 'aeiou')
    alpha = sum(1 for char in text if char.isalpha())
    if alpha > 0 and (vowels / alpha) < 0.15:
        return False, "Text contains an unnaturally low number of vowels."
        
    # Check 2: Presence of common English/Job connector words
    common_words = {"and", "to", "the", "of", "for", "in", "with", "experience", "job", "work", "we", "are", "looking", "is", "this", "will"}
    word_list = re.findall(r'\b[a-z]{2,}\b', text.lower())
    
    # Count how many times any of the common words appear in the text
    common_word_count = sum(1 for w in word_list if w in common_words)
    
    # If there are at least a handful of words, demand at least TWO common connector words
    if len(word_list) > 5 and common_word_count < 2:
        return False, "Text completely lacks basic English vocabulary structure (requires at least 2 common words)."
        
    return True, ""

def predict_job(text, model, vectorizer):
    """Transforms text and predicts if it is a scam or genuine using a specified model."""
    try:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        # Get probability of scam (class 1)
        proba = model.predict_proba(text_vec)[0]
        confidence = proba[prediction]
        
        is_scam = bool(prediction == 1)
        return is_scam, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def highlight_suspicious_keywords(text, vectorizer=None, top_n=5):
    """Finds and wraps suspicious keywords in HTML for highlighting and extracts top TF-IDF features."""
    suspicious_words = [
        r"\burgent\b", r"\bwire text\b", r"\bwire transfer\b", r"\bsocial security\b",
        r"\bbank account\b", r"\bupfront\b", r"\bfive figure\b", r"\bsix figure\b",
        r"\bno experience\b", r"\beasy money\b", r"\bguaranteed\b", r"\bwestern union\b",
        r"\bfee\b", r"\badvance\b", r"\bclick this link\b"
    ]
    
    highlighted_text = text
    found_flags = []
    
    for word_pattern in suspicious_words:
        matches = re.finditer(word_pattern, text, re.IGNORECASE)
        for match in matches:
            word = match.group()
            if word.lower() not in found_flags:
                found_flags.append(word.lower())
            
    # Safer highlight replacement
    for flag in found_flags:
        pattern = re.compile(re.escape(flag), re.IGNORECASE)
        highlighted_text = pattern.sub(f"<span class='highlight-scam'>{flag}</span>", highlighted_text)
        
    top_tfidf_words = []
    if vectorizer is not None:
        try:
            tfidf_matrix = vectorizer.transform([text])
            feature_names = vectorizer.get_feature_names_out()
            dense = tfidf_matrix.todense()
            doc_tfidf = np.array(dense)[0]
            top_indices = doc_tfidf.argsort()[-top_n:][::-1]
            top_tfidf_words = [feature_names[i] for i in top_indices if doc_tfidf[i] > 0]
        except Exception:
            pass
            
    return highlighted_text, found_flags, top_tfidf_words

@st.cache_resource
def get_shap_explainer(_model, model_name, _vectorizer):
    try:
        import shap
        import numpy as np
        if model_name == "Random Forest":
            return shap.TreeExplainer(_model)
        elif model_name == "Logistic Regression":
            # For linear explainers lacking background distributions, use independent zeros baseline
            num_features = len(_vectorizer.get_feature_names_out())
            return shap.LinearExplainer(_model, np.zeros((1, num_features)))
    except Exception as e:
        st.write(f"Explanation init failed: {e}")
        return None
    return None

def get_shap_explanations(text, model, model_name, vectorizer):
    """
    Computes top positive (Scam) and negative (Genuine) SHAP contributing words.
    """
    if model_name == "Naive Bayes":
        # Native SHAP does not safely fast-parse NB without sluggish KernelExplainers
        return [], []
        
    explainer = get_shap_explainer(model, model_name, vectorizer)
    if explainer is None:
        return [], []
        
    try:
        import shap
        X = vectorizer.transform([text])
        if model_name == "Random Forest":
            shap_values = explainer.shap_values(X.toarray())
            if isinstance(shap_values, list): # Common Scikit-learn Random Forest structure (binary class)
                vals = shap_values[1][0]
            else: # Newer SHAP returns
                if len(shap_values.shape) == 3: 
                    vals = shap_values[0, :, 1]
                else: 
                    vals = shap_values[0]
        elif model_name == "Logistic Regression":
            shap_values = explainer.shap_values(X.toarray())
            vals = shap_values[0]
            
        feature_names = vectorizer.get_feature_names_out()
        
        # Sort values
        word_contributions = [(feature_names[i], vals[i]) for i in range(len(vals)) if vals[i] != 0]
        word_contributions.sort(key=lambda x: x[1], reverse=True)
        
        top_scam = [(w, v) for w, v in word_contributions if v > 0][:5]
        
        top_genuine = [(w, v) for w, v in word_contributions if v < 0]
        top_genuine.sort(key=lambda x: x[1])
        top_genuine = top_genuine[:5]
        
        return top_scam, top_genuine
    except Exception as e:
        return [], []
