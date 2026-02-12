"""
ClassifAI - Logistic Regression Email Spam Classifier
A Streamlit web application for real-time email spam classification
"""

import streamlit as st
import joblib
import re
import os

# Page configuration
st.set_page_config(
    page_title="ClassifAI - Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .spam-result {
        background-color: #ffebee;
        color: #c62828;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #c62828;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .ham-result {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .confidence-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer from joblib file"""
    try:
        import joblib
        import os
        
        # Try multiple possible paths for Render deployment
        possible_paths = [
            'Spam_email_classifier.pkl',
            './Spam_email_classifier.pkl',
            os.path.join(os.path.dirname(__file__), 'Spam_email_classifier.pkl'),
            os.path.join(os.getcwd(), 'Spam_email_classifier.pkl')
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            st.error("❌ Model file not found! Please ensure 'Spam_email_classifier.pkl' is in the same directory.")
            st.stop()
        
        # Load using joblib
        model_data = joblib.load(model_path)
        
        # Handle different formats
        if isinstance(model_data, dict):
            model = model_data.get('model')
            vectorizer = model_data.get('vectorizer')
            if vectorizer is None:
                st.error("❌ Vectorizer not found in model file. Please retrain and save the model with the vectorizer together.")
                st.stop()
            return model, vectorizer
        elif isinstance(model_data, tuple):
            model, vectorizer = model_data[0], model_data[1]
            if vectorizer is None:
                st.error("❌ Vectorizer not found in model file. Please retrain and save the model with the vectorizer together.")
                st.stop()
            return model, vectorizer
        else:
            # Assume it's just the model - this is problematic
            st.error("❌ Model file format is invalid. The file should contain both the trained model and the fitted vectorizer. Please retrain the model and save it properly using joblib.save({'model': model, 'vectorizer': vectorizer}, 'Spam_email_classifier.pkl')")
            st.stop()
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'Spam_email_classifier.pkl' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def preprocess_text(text):
    """
    Basic text preprocessing
    Args:
        text (str): Input email text
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def predict_spam(email_text, model, vectorizer):
    """
    Predict if email is spam or not
    Args:
        email_text (str): Input email text
        model: Trained logistic regression model
        vectorizer: Fitted vectorizer
    Returns:
        tuple: (prediction, probability)
    """
    try:
        # Preprocess
        cleaned_text = preprocess_text(email_text)
        
        if not cleaned_text:
            return None, None
        
        # Vectorize
        vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">📧 ClassifAI - Email Spam Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time spam detection using Logistic Regression</div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model..."):
        model, vectorizer = load_model()
    
    # Sidebar - Educational Content
    with st.sidebar:
        st.header("📚 About This App")
        
        st.markdown("""
        ### What is ClassifAI?
        ClassifAI is an educational tool that demonstrates email spam classification 
        using machine learning.
        
        ### How It Works
        1. **Input**: You enter an email text
        2. **Processing**: Text is cleaned and vectorized
        3. **Classification**: Logistic regression predicts spam/not spam
        4. **Output**: See the result with confidence score
        """)
        
        st.markdown("---")
        
        st.header("🧠 About Logistic Regression")
        st.markdown("""
        **Logistic Regression** is a statistical model that:
        - Uses a logistic function to model binary outcomes
        - Calculates probability of an email being spam
        - Works well for text classification tasks
        
        **Why it's effective for spam detection:**
        - Fast training and prediction
        - Interpretable results
        - Good accuracy on text data
        - Low computational requirements
        """)
        
        st.markdown("---")
        
        st.header("🎯 Model Performance")
        st.markdown("""
        **Target Metrics:**
        - Accuracy: ≥ 90%
        - Response Time: ≤ 2 seconds
        - Educational & User-Friendly
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Test Your Email")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Email", "Use Example"],
            horizontal=True
        )
        
        if input_method == "Use Example":
            example_type = st.selectbox(
                "Select example type:",
                ["Spam Example", "Legitimate Email Example"]
            )
            
            if example_type == "Spam Example":
                email_text = st.text_area(
                    "Email Text:",
                    value="WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
                    height=150,
                    help="This is an example of a spam email"
                )
            else:
                email_text = st.text_area(
                    "Email Text:",
                    value="Hey! Are we still meeting for coffee at 3pm today? Let me know if you're running late. See you soon!",
                    height=150,
                    help="This is an example of a legitimate email"
                )
        else:
            email_text = st.text_area(
                "Enter email text to classify:",
                height=200,
                placeholder="Paste or type your email content here...",
                help="Enter the full text of the email you want to check"
            )
        
        # Predict button
        predict_button = st.button("🚀 Classify Email", type="primary", use_container_width=True)
        
        if predict_button:
            if not email_text or len(email_text.strip()) == 0:
                st.error("⚠️ Please enter some email text to classify!")
            else:
                with st.spinner("Analyzing email..."):
                    prediction, probability = predict_spam(email_text, model, vectorizer)
                    
                    if prediction is not None:
                        st.markdown("---")
                        st.subheader("📊 Classification Result")
                        
                        # Display result
                        if prediction == 1:
                            st.markdown(
                                '<div class="spam-result">🚫 SPAM DETECTED</div>',
                                unsafe_allow_html=True
                            )
                            confidence = probability[1] * 100
                            risk_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
                        else:
                            st.markdown(
                                '<div class="ham-result">✅ LEGITIMATE EMAIL</div>',
                                unsafe_allow_html=True
                            )
                            confidence = probability[0] * 100
                            risk_level = "Safe"
                        
                        # Confidence display
                        st.markdown('<div class="confidence-box">', unsafe_allow_html=True)
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence:.2f}%"
                        )
                        
                        # Progress bar
                        st.progress(confidence / 100)
                        
                        # Additional info
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Classification", "Spam" if prediction == 1 else "Not Spam")
                        with col_b:
                            st.metric("Risk Level", risk_level)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Probability breakdown
                        with st.expander("📈 View Probability Breakdown"):
                            prob_col1, prob_col2 = st.columns(2)
                            with prob_col1:
                                st.metric("Spam Probability", f"{probability[1]*100:.2f}%")
                            with prob_col2:
                                st.metric("Ham Probability", f"{probability[0]*100:.2f}%")
    
    with col2:
        st.header("💡 Tips")
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Common Spam Indicators:**
        - 🎁 Prize/lottery claims
        - 💰 Urgent money requests
        - 📱 Premium rate numbers
        - ⚠️ Urgent action required
        - 🔗 Suspicious links
        - 💳 Request for personal info
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Legitimate Email Signs:**
        - 👤 Personal conversation
        - 📅 Meeting reminders
        - 🤝 Normal requests
        - 📧 Professional tone
        - ✅ Expected content
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.header("📝 Quick Stats")
        st.info(f"**Email Length:** {len(email_text)} characters")
        st.info(f"**Word Count:** {len(email_text.split())} words")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>ClassifAI</strong> - Educational Spam Classifier</p>
        <p>Built with Streamlit 🎈 | Powered by Logistic Regression 🤖</p>
        <p style='font-size: 0.8rem;'>⚠️ For educational purposes only. Always verify suspicious emails independently.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
