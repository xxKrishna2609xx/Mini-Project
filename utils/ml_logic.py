import streamlit as st
import joblib
import os
import re

@st.cache_resource
def load_models():
    """Loads and caches the trained model and vectorizer."""
    try:
        if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
            st.error("Model files not found. Please run `setup_mock_model.py` to generate them.")
            return None, None
            
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def predict_job(text, model, vectorizer):
    """Transforms text and predicts if it is a scam or genuine."""
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

def highlight_suspicious_keywords(text):
    """Finds and wraps suspicious keywords in HTML for highlighting."""
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
            
            # Replace exactly the matched string with HTML
            # Note: this is a simple string replace for demo purposes. 
            # Real regex sub should be used carefully to avoid double-wrapping.
            
    # Safer highlight replacement
    for flag in found_flags:
        # Case insensitive replace
        pattern = re.compile(re.escape(flag), re.IGNORECASE)
        highlighted_text = pattern.sub(f"<span class='highlight-scam'>{flag}</span>", highlighted_text)
        
    return highlighted_text, found_flags
