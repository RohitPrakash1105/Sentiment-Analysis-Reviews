# Sentiment-Analysis-Reviews
Sentiment Analysis with BERT
This project is a simple and interactive Sentiment Analysis Web App built using Streamlit and a fine-tuned BERT model.
It allows you to enter any sentence or review, and it will predict whether the sentiment is Positive, Negative, or Neutral.

Live Demo
https://sentiment-analysis-reviewsgit-x2ywxy7yiifr4mtfqg9yuz.streamlit.app/

About the Project
We all have opinions — sometimes positive, sometimes negative, and sometimes in between.
This app uses state-of-the-art NLP (Natural Language Processing) to read your text and guess the sentiment behind it.
The BERT model was fine-tuned on the Amazon Fine Food Reviews dataset, so it’s particularly good at analysing product and service reviews.

Features
Simple, user-friendly UI – just type and get instant predictions.

BERT-powered accuracy – benefits from deep learning on real-world review data.

Runs in your browser – no need for installation on your side (if using hosted link).

Test with your own sentences or try the provided examples.

 How to Use
Open the app in your browser (see Live Demo above).

Type or paste your review/opinion in the input box.
Example: "I absolutely loved the food and the service was amazing!"

Click Predict Sentiment.

The app will display:

Predicted Sentiment (Positive / Negative / Neutral)

A confidence score so you know how sure the model is.

Model Details
Architecture: BERT (Bidirectional Encoder Representations from Transformers)

Fine-tuned Dataset: Amazon Fine Food Reviews

Frameworks Used:

Hugging Face Transformers

PyTorch

 Tech Stack
Python

Streamlit – for the web app

Transformers – for loading the BERT model

Torch – for model inference
