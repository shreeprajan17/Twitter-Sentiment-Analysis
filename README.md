# Twitter Sentiment Analysis

This project performs sentiment analysis on tweets using Natural Language Processing (NLP) techniques. It uses a pre-trained model to classify tweets as either positive or negative, with a graphical user interface (GUI) created using Streamlit for easy interaction.

## Project Overview

The goal of this project is to predict the sentiment of a tweet (positive or negative) based on its content. The project utilizes the **Sentiment140** dataset for training and testing a logistic regression model, combined with TF-IDF vectorization for text preprocessing. Users can input a tweet through the GUI, and the model will output the predicted sentiment along with an accuracy score.

## Features

- Sentiment prediction for individual tweets.
- A simple and intuitive web-based user interface using **Streamlit**.
- Preprocessing steps to clean and normalize tweet data (removing mentions, URLs, and special characters).
- Visualization of sentiment distribution through word clouds (generated in Google Colab).

## Screenshots
![Positive Tweet](positive_tweet_output.png)
![Negative Tweet](negative_tweet_output.png)

## Project Structure

- `sentiment_model.pkl`: Pre-trained logistic regression model for sentiment classification.
- `tfidf_vectorizer.pkl`: Trained TF-IDF vectorizer for text transformation.
- `app.py`: Streamlit app file for running the sentiment analyzer.
- `twittersentimentanalysis.ipynb`: Jupyter notebook (Google Colab) used for model training and data preprocessing.
- `README.md`: Project documentation (this file).

## Installation

### Prerequisites
- Python 3.x
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `nltk`, `joblib`, `matplotlib`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. After running the Streamlit app, you can enter a tweet in the text box to analyze its sentiment.
2. The app will display whether the tweet is positive or negative, along with the accuracy of the prediction.

## Dataset

The model was trained on the **Sentiment140** dataset, which contains 1.6 million labeled tweets. The dataset is publicly available and widely used for sentiment analysis tasks.

## Future Enhancements

- **Multi-language support** for tweets in different languages.
- **Real-time tweet analysis** using the Twitter API.
- **Improved model performance** using more advanced NLP techniques.
