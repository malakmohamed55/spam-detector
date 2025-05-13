import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the model and vectorizer
@st.cache_resource
def load_model():
    with open('spam_classifier.pkl', 'rb') as f:
        vectorizer, classifier = pickle.load(f)
    return vectorizer, classifier

vectorizer, classifier = load_model()

# Text preprocessing function (same as in your notebook)
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit app
st.title("Spam Message Classifier")
st.write("This app predicts whether a text message is spam or not (ham).")

# Input text
user_input = st.text_area("Enter a message to classify:", "")

if st.button("Classify"):
    if user_input:
        # Preprocess the input
        processed_text = preprocess_text(user_input)
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = classifier.predict(text_vector)
        prediction_proba = classifier.predict_proba(text_vector)
        
        # Display results
        if prediction[0] == 1:
            st.error("This message is **SPAM** (confidence: {:.2f}%)".format(prediction_proba[0][1] * 100))
        else:
            st.success("This message is **HAM** (not spam) (confidence: {:.2f}%)".format(prediction_proba[0][0] * 100))
        
        # Show probability breakdown
        st.write("**Probability breakdown:**")
        st.write(f"- Ham (not spam): {prediction_proba[0][0] * 100:.2f}%")
        st.write(f"- Spam: {prediction_proba[0][1] * 100:.2f}%")
    else:
        st.warning("Please enter a message to classify.")

#streamlit run spam_detector_app.py