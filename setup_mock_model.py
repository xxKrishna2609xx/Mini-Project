import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def generate_mock_data(n_samples=500):
    # Genuine keywords
    genuine_titles = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Intern", "HR Executive"]
    genuine_desc = ["We are looking for a talented individual to join our growing team. Responsibilities include building scalable systems.", 
                    "Join our fast-paced startup. Competitive salary and comprehensive benefits.",
                    "Seeking a recent graduate to help with daily operations and learn from our senior staff.",
                    "Work with our core product team to deliver high-quality software.",
                    "Responsible for managing end-to-end marketing campaigns."]
    
    # Scam keywords
    scam_titles = ["Data Entry Clerk - URgENT", "Work From Home Make $5000/week", "Virtual Assistant Immediate Hire", "Package Forwarder", "Secret Shopper"]
    scam_desc = ["Earn money from home with no experience needed! Start immediately! Wire advance fee to begin.",
                 "Provide your bank details to receive the first paycheck. Urgent hiring.",
                 "Easy money, work just 1 hour a day. Please send social security number for background check.",
                 "We guarantee a six figure salary without any prior experience. Pay processing fee upfront.",
                 "Click this link to claim your sign on bonus. Only suitable for quick learners."]
    
    data = []
    
    for _ in range(n_samples):
        is_scam = np.random.choice([0, 1], p=[0.7, 0.3]) # 30% scam
        if is_scam:
            title = np.random.choice(scam_titles)
            desc = np.random.choice(scam_desc)
            text = f"{title} {desc}"
        else:
            title = np.random.choice(genuine_titles)
            desc = np.random.choice(genuine_desc)
            text = f"{title} {desc}"
            
        data.append({
            "title": title,
            "description": desc,
            "text": text,
            "is_scam": is_scam
        })
        
    df = pd.DataFrame(data)
    df.to_csv("fake_job_postings.csv", index=False)
    print("Created fake_job_postings.csv")
    return df

def train_and_save_model():
    print("Generating mock data...")
    df = generate_mock_data()
    
    print("Training mock models...")
    X = df["text"]
    y = df["is_scam"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train Model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Mock Model Accuracy: {acc:.2f}")
    
    # Save
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(model, "model.pkl")
    print("Saved vectorizer.pkl and model.pkl")

if __name__ == "__main__":
    train_and_save_model()
    print("Setup complete! You can now run `streamlit run app.py`.")
