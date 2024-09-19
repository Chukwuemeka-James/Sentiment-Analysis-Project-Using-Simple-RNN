# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit App Design
st.markdown("<h1 style='text-align: center; color: #FF5733;'>IMDB Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("This app classifies a movie review as **positive** or **negative** based on the sentiment.")

# Subheader for user input
st.subheader('Enter a Movie Review:')

# User input text area
user_input = st.text_area('Type your review here...', height=150)

# Classification button
if st.button('Classify'):
    
    with st.spinner('Analyzing the review...'):
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.markdown(f"<h3 style='color: #4CAF50;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
        st.write(f'**Prediction Confidence Score:** {prediction[0][0]:.2f}')
else:
    st.write('Please enter a movie review.')

# Add a footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Built with 💻 and ☕ by [Your Name]</p>", unsafe_allow_html=True)
