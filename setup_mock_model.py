import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import json
def generate_mock_data(n_samples=500):
    # Genuine keywords
    genuine_titles = [
        "Software Engineer", "Data Scientist", "Product Manager", "Marketing Intern", "HR Executive",
        "Frontend Developer", "Backend Engineer", "Full Stack Developer", "DevOps Engineer", "Cloud Architect",
        "Machine Learning Engineer", "Data Analyst", "Business Analyst", "Marketing Manager", "Sales Representative",
        "Customer Success Manager", "Account Executive", "Graphic Designer", "UX/UI Designer", "Financial Analyst",
        "Project Manager", "Operations Manager", "Human Resources Manager", "Content Writer", "Social Media Manager",
        "Network Engineer", "Systems Administrator", "Database Administrator", "Information Security Analyst",
        "Technical Support Specialist", "Quality Assurance Tester", "Legal Counsel", "Registered Nurse", "Teacher",
        "Administrative Assistant", "Executive Assistant", "Office Manager", "Research Scientist", "Biomedical Engineer"
    ]
    
    genuine_desc = [
        "We are looking for a talented individual to join our growing team. Responsibilities include building scalable systems.", 
        "Join our fast-paced startup. Competitive salary and comprehensive benefits.",
        "Seeking a recent graduate to help with daily operations and learn from our senior staff.",
        "Work with our core product team to deliver high-quality software.",
        "Responsible for managing end-to-end marketing campaigns.",
        "Looking for an experienced professional to lead our engineering efforts and mentor junior developers.",
        "You will be responsible for designing and implementing RESTful APIs using Python and Django.",
        "Seeking a highly motivated individual with a passion for data analysis and machine learning.",
        "This role involves collaborating with cross-functional teams to define, design, and ship new features.",
        "We offer a dynamic work environment with opportunities for professional growth and development.",
        "The ideal candidate will have strong problem-solving skills and a solid understanding of software design patterns.",
        "Responsible for maintaining and optimizing our cloud infrastructure on AWS.",
        "You will be analyzing complex datasets to extract actionable insights for business strategy.",
        "Seeking a creative designer to develop engaging user interfaces for our web and mobile applications.",
        "Manage client relationships and ensure customer satisfaction throughout the project lifecycle.",
        "Requirements include a Bachelor's degree in a related field and 3+ years of professional experience."
    ]
    
    # Scam keywords
    scam_titles = [
        "Data Entry Clerk - URgENT", "Work From Home Make $5000/week", "Virtual Assistant Immediate Hire", 
        "Package Forwarder", "Secret Shopper", "Online Evaluator", "Mystery Shopper Needed", "Typing Jobs From Home",
        "Envelope Stuffer", "Re-shipping Clerk", "Payment Processor", "Assembly Work At Home", "Email Processor",
        "Customer Service Representative - NO EXPERIENCE", "Administrative Assistant - Easy Money", 
        "Make Money Online Fast", "Guaranteed Income Part Time", "Work From Home Typist", "Financial Agent",
        "Account Manager - No Background Check", "Remote Data Entry - $35/hr", "Customer Support - Start Today",
        "Data Analyst (Entry Level) - Easy Apply", "Payroll Assistant - Remote", "General Labor - Cash Pay"
    ]
    
    scam_desc = [
        "Earn money from home with no experience needed! Start immediately! Wire advance fee to begin.",
        "Provide your bank details to receive the first paycheck. Urgent hiring.",
        "Easy money, work just 1 hour a day. Please send social security number for background check.",
        "We guarantee a six figure salary without any prior experience. Pay processing fee upfront.",
        "Click this link to claim your sign on bonus. Only suitable for quick learners.",
        "You have been selected for this position! Please send us your full name, address, and bank account for verification to get started.",
        "Work remotely and earn up to $500 a day! Simply pay a small software license fee and start processing emails.",
        "We will send you a check for $2000. Keep your $500 salary and wire transfer the remaining $1500 to our supplier.",
        "URGENT: We need to hire 50 people by tomorrow. No interview required. Just pay the background check fee.",
        "This is a legitimate work from home opportunity. 100% Guaranteed income. Reply with your resume and bank details.",
        "Get paid to shop and rate customer service. We will advance you the funds, just deposit this check.",
        "Easy data entry tasks. You must purchase your own equipment through our approved vendor after we deposit a check into your account.",
        "Make thousands a week stuffing envelopes! Send a $50 registration fee for the starter kit.",
        "No skills necessary. We train you. Just provide your personal information including social security for tax purposes.",
        "Immediate opening! To apply, click the link and enter your credit card information for identity verification."
    ]
    
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
            "fraudulent": is_scam
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
    y = df["fraudulent"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Naive Bayes": MultinomialNB()
    }
    
    metrics = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()
        
        metrics[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm
        }
        
        # Save model
        filename = ""
        if name == "Random Forest":
            filename = "rf_model.pkl"
        elif name == "Logistic Regression":
            filename = "lr_model.pkl"
        else:
            filename = "nb_model.pkl"
            
        joblib.dump(model, filename)
        print(f"Saved {filename}")

    # Save Vectorizer
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved vectorizer.pkl")
    
    # Save Metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics.json")

if __name__ == "__main__":
    train_and_save_model()
    print("Setup complete! You can now run `streamlit run app.py`.")
