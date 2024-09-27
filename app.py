import nltk
import streamlit as st
import joblib
import re
import sklearn
from nltk.corpus import stopwords

# Loading pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Function to preprocess the tweet
def preprocess_new_tweet(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text


# Function to predict sentiment
def predict_sentiment(tweet):
    cleaned_tweet = preprocess_new_tweet(tweet)
    tweet_tfidf = vectorizer.transform([cleaned_tweet])
    prediction = model.predict(tweet_tfidf)
    return "Positive" if prediction == 1 else "Negative"


# Streamlit UI
st.title("Tweet Sentiment Analyzer")
tweet = st.text_input("Enter a tweet to analyze sentiment:")

if st.button("Predict Sentiment"):
    if tweet:
        sentiment = predict_sentiment(tweet)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a tweet.")

