# IMDB Movie Review Sentiment Analysis with Simple RNN

This project demonstrates an end-to-end sentiment analysis on the IMDB movie review dataset using a Simple Recurrent Neural Network (Simple RNN). The project includes loading the dataset, building the model, training it, and deploying a web app for users to classify movie reviews as positive or negative.

## Project Structure

- `main.py`: Contains the Streamlit app code that accepts a user input review and classifies it.
- `train_model.py`: Trains the Simple RNN on the IMDB dataset and saves the model.
- `simple_rnn_imdb.h5`: The pre-trained model used by the Streamlit app for sentiment prediction.


## Dataset

The IMDB dataset consists of 50,000 movie reviews, divided equally into training and testing sets. Each review is encoded as a sequence of integers representing the words, and the model classifies the review as either positive or negative.

- **Training data shape**: 25,000 reviews
- **Testing data shape**: 25,000 reviews

The dataset is loaded from the `tensorflow.keras.datasets` library.


## Steps

### 1. Model Training

I train a Simple RNN model on the IMDB dataset with an embedding layer and a single Simple RNN layer followed by a Dense layer for binary classification. Early stopping is used to prevent overfitting. The model is trained for 10 epochs, and early stopping is applied when the validation loss does not improve after 5 epochs.

The trained model is saved as `simple_rnn_imdb.h5`.

### 2. Streamlit App

The Streamlit app allows users to input a movie review, preprocess it, and then classify the sentiment using the pre-trained model. The review is tokenized, padded, and passed to the model for prediction.

Key parts of the app include:

- **Preprocessing the review**: The user input is tokenized and padded to ensure it's compatible with the trained model.
- **Model Prediction**: The pre-trained model classifies the review as either "Positive" or "Negative" based on the prediction score.

### 3. Helper Functions

- **`decode_review`**: Converts the encoded review back to readable text.
- **`preprocess_text`**: Converts the user input into the correct format for prediction.
- **`predict_sentiment`**: Runs the model on preprocessed input and returns the predicted sentiment.


## Conclusion

This project demonstrates how to train a simple RNN model on text data and deploy it using a Streamlit web app. It allows users to classify movie reviews into positive or negative sentiments using a pre-trained model.