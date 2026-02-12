"""
Training script for ClassifAI - Email Spam Classifier
This script trains a Logistic Regression model with a fitted vectorizer
and saves both to a pickle file.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Sample SMS/Email data for training (can be replaced with actual dataset)
training_data = [
    # Spam examples
    ("WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.", 1),
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to enter 100% GUARANTEED for å£/å£1 wd eachweek TandC www.footballfanatics.co.uk JPLS claimyourreward of 3pounds pomontevideo", 1),
    ("U HAVE WON A CASH PRIZE! C ALL 09061701461 CLAIM UR PRIZE', 'How To Get Home & Car Repair GUARANTEED!!! Free Quote Online Now!!! Call Now 1-800-333-3171 Serious Responses Only!", 1),
    ("Congratulations! You've won. Claim your prize worth $5000 here: www.spamsite.com", 1),
    ("URGENT: Your account will be closed due to suspicious activity. Click here to verify: www.fake-bank.com", 1),
    ("LIMITED TIME: 50% off everything! Act now before offer expires. www.marketingscam.com", 1),
    ("You have been selected for a special offer. Click here to claim your reward", 1),
    ("DEAR SIR/MADAM, I have a business proposal for you involving millions of dollars", 1),
    ("Get rich quick! Work from home and earn $5000/week guaranteed!", 1),
    ("Click here to download your free gift card worth $100", 1),
    
    # Legitimate email examples
    ("Hey! Are we still meeting for coffee at 3pm today? Let me know if you're running late. See you soon!", 0),
    ("Hi John, I wanted to follow up on our meeting yesterday. Can we schedule another call next week?", 0),
    ("Your package has been delivered. Tracking number: 1234567890", 0),
    ("Meeting reminder: Project sync at 2pm in Conference Room B", 0),
    ("Thanks for your email. I'll get back to you on this by end of week.", 0),
    ("Hi Sarah, How are you doing? Hope to catch up soon over coffee!", 0),
    ("FYI: The new project timeline has been updated. Check the shared drive for details.", 0),
    ("Hi, I wanted to invite you to our team lunch on Friday at noon. RSVP please?", 0),
    ("Just confirming our meeting tomorrow at 10am in Building A?", 0),
    ("Thanks for sharing the files. I've reviewed them and have some feedback.", 0),
    ("Hi All, The presentation slides have been uploaded to the team folder.", 0),
    ("Can you please send me the updated report when you get a chance?", 0),
]

def train_spam_classifier():
    """Train the spam classifier model and save it with the vectorizer"""
    
    print("=" * 60)
    print("Training Spam Classifier Model")
    print("=" * 60)
    
    # Extract texts and labels
    texts = [data[0] for data in training_data]
    labels = [data[1] for data in training_data]
    
    print(f"\nDataset: {len(texts)} samples")
    print(f"  - Spam: {sum(labels)}")
    print(f"  - Legitimate: {len(labels) - sum(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and fit vectorizer
    print("\n[1/3] Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    # Train model
    print("[2/3] Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_vectorized, y_train)
    print("Model trained successfully!")
    
    # Evaluate
    print("[3/3] Evaluating model...")
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("Model Performance Metrics")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 60)
    
    # Save model and vectorizer together
    print("\nSaving model and vectorizer...")
    model_data = {
        'model': model,
        'vectorizer': vectorizer
    }
    
    joblib.dump(model_data, 'Spam_email_classifier.pkl')
    print("✅ Model saved successfully as 'Spam_email_classifier.pkl'")
    print("\nThe model file contains:")
    print("  - Trained Logistic Regression model")
    print("  - Fitted TfidfVectorizer with vocabulary")
    print("\n✅ Ready to use with app.py!")
    
    return model, vectorizer

if __name__ == "__main__":
    train_spam_classifier()
