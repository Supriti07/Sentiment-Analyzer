#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset and preprocess as before
@st.cache
def load_data():
    sentiment_data = pd.read_csv('Tweets.csv')
    sentiment_df = sentiment_data.drop(sentiment_data[sentiment_data['airline_sentiment_confidence'] < 0.5].index, axis=0)
    return sentiment_df

def clean_text(text):
    stop_words = stopwords.words('english')
    punctuations = string.punctuation
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if (word not in stop_words) and (word not in punctuations)]
    text = ' '.join(text)
    return text

def preprocess_data(sentiment_df):
    X = sentiment_df['text']
    Y = sentiment_df['airline_sentiment']
    
    clean_data = [clean_text(text) for text in X]
    sentiments = ['negative', 'neutral', 'positive']
    Y = Y.apply(lambda x: sentiments.index(x))

    count_vectorizer = CountVectorizer(max_features=5000, stop_words=['virginamerica', 'united'])
    X_fit = count_vectorizer.fit_transform(clean_data).toarray()
    
    return X_fit, Y, count_vectorizer

# Load and preprocess data
sentiment_df = load_data()
X_fit, Y, count_vectorizer = preprocess_data(sentiment_df)

# Train model
model = MultinomialNB()
X_train, X_test, Y_train, Y_test = train_test_split(X_fit, Y, test_size=0.3, random_state=42)
model.fit(X_train, Y_train)

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(count_vectorizer, 'count_vectorizer.pkl')

# Streamlit app
st.title("Sentiment Analysis of Airline Tweets")

# User input
user_input = st.text_area("Enter a tweet for sentiment analysis")

if st.button("Analyze Sentiment"):
    if user_input:
        clean_input = clean_text(user_input)
        vectorizer = joblib.load('count_vectorizer.pkl')
        model = joblib.load('sentiment_model.pkl')
        X_input = vectorizer.transform([clean_input]).toarray()
        prediction = model.predict(X_input)
        
        sentiments = ['Negative', 'Neutral', 'Positive']
        sentiment_result = sentiments[prediction[0]]
        
        st.write(f"Sentiment: {sentiment_result}")
    else:
        st.write("Please enter a tweet for analysis.")

# Display accuracy and classification report
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
classification = classification_report(Y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.write("Classification Report:")
st.text(classification)
